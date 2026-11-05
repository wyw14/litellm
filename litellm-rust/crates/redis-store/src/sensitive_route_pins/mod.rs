mod keys;

use std::time::Duration;

use litellm_python_compat::{Value, json};
use litellm_redis_client::{Ttl, commands};

use crate::{
    DecodeError, Error,
    codec::python_value,
    memory::{Clock, Memory, system_clock},
    remote::{Remote, RemoteSlot},
    value::text,
};

/// The model each session is pinned to: memory per worker, Redis shared by every pod.
pub struct SensitiveRoutePins {
    pins: Memory<String>,
    remote: RemoteSlot,
}

impl SensitiveRoutePins {
    pub fn new(remote: Option<Remote>) -> Self {
        Self::with_clock(remote, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, now: Clock) -> Self {
        Self {
            pins: Memory::new(None, &now),
            remote: RemoteSlot::new(remote),
        }
    }

    /// Redis first, kept in memory for the key's remaining TTL. A Redis miss or failure serves memory.
    pub async fn get(
        &self,
        tenant: &str,
        session_id: &str,
        default_ttl: Duration,
    ) -> Result<Option<String>, Error> {
        let key = keys::pin(tenant, session_id);
        if let Some(model) = self.read_remote(&key, default_ttl).await? {
            return Ok(Some(model));
        }
        Ok(self.pins.get(&key)?)
    }

    /// `SET ... EX` whose failure only logs, then memory.
    pub async fn set(
        &self,
        tenant: &str,
        session_id: &str,
        model: &str,
        ttl: Duration,
    ) -> Result<(), Error> {
        let key = keys::pin(tenant, session_id);
        if let Some(client) = self.remote.client() {
            let encoded = json::dumps(&Value::Str(model.to_owned()))?;
            if let Err(error) = client.send(commands::set(&key, encoded, ttl)).await {
                tracing::warn!(%error, "writing a sensitive route pin to Redis failed");
            }
        }
        self.pins.set(&key, model.to_owned(), Some(ttl))?;
        Ok(())
    }

    /// `GET`, then on a hit its own `TTL` trip, as `RedisCache.async_get_cache` and `async_get_ttl`
    /// make them. A key without a TTL, or a failed `TTL`, keeps `default_ttl`.
    async fn read_remote(&self, key: &str, default_ttl: Duration) -> Result<Option<String>, Error> {
        let Some(client) = self.remote.client() else {
            return Ok(None);
        };
        let read = client
            .send(commands::get::<Vec<u8>>(key))
            .await
            .map_err(Error::from)
            .and_then(|raw| raw.as_deref().map(decode).transpose().map_err(Error::from));
        let model = match read {
            Ok(Some(model)) => model,
            Ok(None) => return Ok(None),
            Err(error) => {
                tracing::warn!(%error, "reading a sensitive route pin from Redis failed, serving memory");
                return Ok(None);
            }
        };
        let ttl = match client.send(commands::ttl(key)).await {
            Ok(Ttl::Expires(remaining)) => remaining,
            Ok(Ttl::Missing | Ttl::Persistent) => default_ttl,
            Err(error) => {
                tracing::debug!(%error, "Redis TTL Error");
                default_ttl
            }
        };
        self.pins.set(key, model.clone(), Some(ttl))?;
        Ok(Some(model))
    }
}

fn decode(raw: &[u8]) -> Result<String, DecodeError> {
    text(&python_value(raw)?, "a pinned model name")
}

//! Pluggable signing for production MM deployments.
//!
//! Most production market-makers do **not** hold a raw private key in the
//! bot's process memory. Instead they sit behind an HSM, a KMS-backed
//! remote signer, MPC threshold-signing, or a hosted custodian service
//! (`Web3Auth`, Privy, Turnkey, Fireblocks). [`DeadeyeSigner`] is the
//! narrow trait that abstracts over all of those.
//!
//! ## Concrete implementations
//!
//! * [`LocalSigner`] — wraps a `starknet_signers::LocalWallet`. The default for
//!   devnet, testnet, and any deployment willing to hold the key in process
//!   memory.
//! * [`RemoteSigner`] — POSTs a Stark hash to an HTTP endpoint and parses the
//!   returned `(r, s)`. Drop-in for HSM gateways or hosted-key services.
//! * [`SimulationSigner`] — never signs and marks itself interactive so
//!   provider-backed skip-validation simulations can be built for address-only
//!   workflows.
//!
//! ## Wiring into [`crate::OwnedAccount`]
//!
//! Pass any `DeadeyeSigner` to [`crate::OwnedAccount::with_signer`].
//! Existing call-sites that use [`crate::OwnedAccount::from_signing_key`]
//! continue to work — internally that constructor wraps the felt in a
//! [`LocalSigner`].
//!
//! ## Sync-shim choice
//!
//! `starknet_signers::Signer` is *already async* in the upstream
//! `starknet-rs` crate (version 0.14+). Our [`DeadeyeSigner`] is also
//! async (`async_trait`), so the adapter that bridges the two is a
//! straight delegating wrapper — **no `block_on` shim, no synchronous
//! `tokio::Handle::block_on` jail**. Each remote-signer call awaits
//! cleanly inside the same runtime that submitted the transaction.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use starknet_core::{crypto::Signature, types::Felt};
use thiserror::Error;

/// The single-method trait every production signing backend implements.
///
/// Implementations must be `Send + Sync` so they can be parked behind an
/// `Arc` and shared across the bot's task graph.
#[async_trait]
pub trait DeadeyeSigner: Send + Sync + core::fmt::Debug {
    /// Stark public-key derivation. The corresponding account address is
    /// computed from this scalar via the OZ deployer pre-computation.
    async fn public_key(&self) -> Result<Felt, SignerError>;

    /// Sign `hash` and return the canonical `(r, s)` pair.
    async fn sign_hash(&self, hash: Felt) -> Result<[Felt; 2], SignerError>;

    /// Whether the underlying signer is "interactive" / expensive — used
    /// by `starknet-accounts` to decide whether to re-sign during a fee
    /// estimation step. Most production signers (HSM/KMS) want this set
    /// to `true` so we don't make extra HSM round-trips when estimating
    /// fees.
    fn is_interactive(&self) -> bool {
        false
    }
}

/// Errors produced by a [`DeadeyeSigner`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SignerError {
    /// The remote signing service refused the request or returned a
    /// transport-level error.
    #[error("transport error: {0}")]
    Transport(String),

    /// The remote service returned a payload we couldn't decode.
    #[error("decode error: {0}")]
    Decode(String),

    /// The signing key returned an ECDSA-level failure (e.g. invalid
    /// `k`). Practically impossible with a healthy CSPRNG but surfaced
    /// for completeness.
    #[error("ecdsa error: {0}")]
    Ecdsa(String),

    /// Configuration is invalid (e.g. empty endpoint URL).
    #[error("invalid config: {0}")]
    Config(String),
}

// ─── LocalSigner
// ──────────────────────────────────────────────────────────────

/// In-process signer backed by `starknet_signers::LocalWallet`.
///
/// Use for devnet, testnets, or deployments where holding the key in the
/// bot's address space is acceptable. Production deployments should
/// prefer [`RemoteSigner`].
#[derive(Debug, Clone)]
pub struct LocalSigner {
    wallet: starknet_signers::LocalWallet,
}

impl LocalSigner {
    /// Build a signer from a raw private-key felt.
    #[must_use]
    pub fn from_signing_key(secret_scalar: Felt) -> Self {
        let key = starknet_signers::SigningKey::from_secret_scalar(secret_scalar);
        Self {
            wallet: starknet_signers::LocalWallet::from_signing_key(key),
        }
    }

    /// Build a signer from an existing `LocalWallet`.
    #[must_use]
    pub const fn from_wallet(wallet: starknet_signers::LocalWallet) -> Self {
        Self { wallet }
    }

    /// Borrow the underlying wallet.
    #[must_use]
    pub const fn wallet(&self) -> &starknet_signers::LocalWallet {
        &self.wallet
    }
}

#[async_trait]
impl DeadeyeSigner for LocalSigner {
    async fn public_key(&self) -> Result<Felt, SignerError> {
        use starknet_signers::Signer;
        let vk = self
            .wallet
            .get_public_key()
            .await
            .map_err(|e| SignerError::Ecdsa(format!("{e}")))?;
        Ok(vk.scalar())
    }

    async fn sign_hash(&self, hash: Felt) -> Result<[Felt; 2], SignerError> {
        use starknet_signers::Signer;
        let sig = self
            .wallet
            .sign_hash(&hash)
            .await
            .map_err(|e| SignerError::Ecdsa(format!("{e}")))?;
        Ok([sig.r, sig.s])
    }

    fn is_interactive(&self) -> bool {
        false
    }
}

// ─── SimulationSigner
// ─────────────────────────────────────────────────────────

/// Signer for address-only simulation paths.
///
/// Starknet transaction simulation with `skip_validate = true` can omit the
/// signature entirely. This signer advertises itself as interactive so
/// `starknet-accounts` skips signing in that mode, while any accidental real
/// signing attempt fails loudly.
#[derive(Debug, Clone, Copy)]
pub struct SimulationSigner {
    public_key: Felt,
}

impl SimulationSigner {
    /// Construct a simulation signer with a placeholder public key.
    #[must_use]
    pub const fn new(public_key: Felt) -> Self {
        Self { public_key }
    }
}

#[async_trait]
impl DeadeyeSigner for SimulationSigner {
    async fn public_key(&self) -> Result<Felt, SignerError> {
        Ok(self.public_key)
    }

    async fn sign_hash(&self, _hash: Felt) -> Result<[Felt; 2], SignerError> {
        Err(SignerError::Config(
            "simulation signer cannot sign real transactions".into(),
        ))
    }

    fn is_interactive(&self) -> bool {
        true
    }
}

// ─── RemoteSigner
// ─────────────────────────────────────────────────────────────

/// Tunables for [`RemoteSigner`].
#[derive(Debug, Clone)]
pub struct RemoteSignerConfig {
    /// Per-call HTTP timeout.
    pub timeout: Duration,
    /// Optional bearer token. When set, the value is prepended with
    /// `Bearer ` and placed in the `Authorization` header.
    pub auth_token: Option<String>,
    /// Number of additional attempts on transient failures (5xx / network
    /// errors / timeout). `0` means "submit once and bail".
    pub max_retries: u32,
    /// Initial backoff between retries. Doubles each retry up to a soft
    /// cap of `8 × backoff`.
    pub backoff: Duration,
}

impl Default for RemoteSignerConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3),
            auth_token: None,
            max_retries: 3,
            backoff: Duration::from_millis(100),
        }
    }
}

/// HTTP-based signer that delegates `sign_hash` to a remote service.
///
/// ## Wire format
///
/// **Request**: `POST <endpoint>` with body
/// ```json
/// {"hash": "0xabcdef..."}
/// ```
/// **Response**: HTTP 200 with body
/// ```json
/// {"r": "0x...", "s": "0x..."}
/// ```
/// Non-2xx status codes are treated as transient (retry-eligible). 4xx
/// codes are still retried — most HSM front-ends rate-limit with `429`
/// and recover quickly.
#[derive(Debug, Clone)]
pub struct RemoteSigner {
    public_key: Felt,
    endpoint: url::Url,
    client: reqwest::Client,
    config: RemoteSignerConfig,
}

impl RemoteSigner {
    /// Construct a `RemoteSigner`.
    ///
    /// The remote service's Stark public key must be known at
    /// construction time — it's used both for account-address
    /// computation and for the `DeadeyeSigner::public_key` impl. Passing
    /// the wrong value will yield signatures that don't verify against
    /// the on-chain account.
    pub fn new(
        public_key: Felt,
        endpoint: url::Url,
        config: RemoteSignerConfig,
    ) -> Result<Self, SignerError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| SignerError::Config(format!("reqwest client: {e}")))?;
        Ok(Self {
            public_key,
            endpoint,
            client,
            config,
        })
    }

    /// Build with a pre-existing reqwest client (for connection pooling
    /// across multiple signers, custom resolvers, etc.).
    pub const fn with_client(
        public_key: Felt,
        endpoint: url::Url,
        client: reqwest::Client,
        config: RemoteSignerConfig,
    ) -> Self {
        Self {
            public_key,
            endpoint,
            client,
            config,
        }
    }

    /// Endpoint URL.
    #[must_use]
    pub const fn endpoint(&self) -> &url::Url {
        &self.endpoint
    }

    async fn sign_once(&self, hash: Felt) -> Result<[Felt; 2], SignerError> {
        let hash_hex = felt_to_hex(hash);
        let body = serde_json::json!({ "hash": hash_hex });
        let mut req = self.client.post(self.endpoint.clone()).json(&body);
        if let Some(token) = self.config.auth_token.as_ref() {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| SignerError::Transport(format!("{e}")))?;
        if !resp.status().is_success() {
            return Err(SignerError::Transport(format!(
                "non-success status: {}",
                resp.status(),
            )));
        }
        let parsed: SignResponse = resp
            .json()
            .await
            .map_err(|e| SignerError::Decode(format!("{e}")))?;
        let r = hex_to_felt(&parsed.r)?;
        let s = hex_to_felt(&parsed.s)?;
        Ok([r, s])
    }
}

#[async_trait]
impl DeadeyeSigner for RemoteSigner {
    async fn public_key(&self) -> Result<Felt, SignerError> {
        Ok(self.public_key)
    }

    async fn sign_hash(&self, hash: Felt) -> Result<[Felt; 2], SignerError> {
        let max_attempts = self.config.max_retries.saturating_add(1);
        let mut backoff = self.config.backoff;
        let mut last_err: Option<SignerError> = None;
        for attempt in 0..max_attempts {
            match self.sign_once(hash).await {
                Ok(rs) => return Ok(rs),
                Err(e) => {
                    last_err = Some(e);
                    if attempt + 1 < max_attempts {
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(self.config.backoff * 8);
                    }
                },
            }
        }
        Err(last_err.unwrap_or_else(|| SignerError::Transport("unknown".into())))
    }

    fn is_interactive(&self) -> bool {
        // Remote signers should be treated as expensive — skip the
        // extra fee-estimation re-sign round-trip.
        true
    }
}

#[derive(Debug, serde::Deserialize)]
struct SignResponse {
    r: String,
    s: String,
}

fn felt_to_hex(f: Felt) -> String {
    let bytes = f.to_bytes_be();
    format!("0x{}", hex::encode(bytes))
}

fn hex_to_felt(s: &str) -> Result<Felt, SignerError> {
    Felt::from_hex(s).map_err(|e| SignerError::Decode(format!("hex {s:?}: {e}")))
}

// ─── Adapter to starknet_signers::Signer
// ──────────────────────────────────────

/// Adapter that wraps a `DeadeyeSigner` and implements
/// `starknet_signers::Signer` so it can be used by
/// `starknet_accounts::SingleOwnerAccount`.
///
/// This adapter is what makes the trait pluggable from inside
/// [`crate::OwnedAccount`] — `SingleOwnerAccount` expects a generic
/// `S: Signer`, so we monomorphize over this adapter type.
#[derive(Debug, Clone)]
pub struct SignerAdapter {
    inner: Arc<dyn DeadeyeSigner>,
}

impl SignerAdapter {
    /// Construct an adapter from any boxed [`DeadeyeSigner`].
    #[must_use]
    pub fn new(inner: Arc<dyn DeadeyeSigner>) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped signer trait object.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn DeadeyeSigner> {
        &self.inner
    }
}

#[async_trait]
impl starknet_signers::Signer for SignerAdapter {
    type GetPublicKeyError = AdapterError;
    type SignError = AdapterError;

    async fn get_public_key(
        &self,
    ) -> Result<starknet_signers::VerifyingKey, Self::GetPublicKeyError> {
        let scalar = self.inner.public_key().await.map_err(AdapterError::from)?;
        Ok(starknet_signers::VerifyingKey::from_scalar(scalar))
    }

    async fn sign_hash(&self, hash: &Felt) -> Result<Signature, Self::SignError> {
        let [r, s] = self
            .inner
            .sign_hash(*hash)
            .await
            .map_err(AdapterError::from)?;
        Ok(Signature { r, s })
    }

    fn is_interactive(&self, _ctx: starknet_signers::SignerInteractivityContext<'_>) -> bool {
        self.inner.is_interactive()
    }
}

/// Adapter-side error wrapping a [`SignerError`].
#[derive(Debug, Error)]
#[error(transparent)]
pub struct AdapterError(SignerError);

impl From<SignerError> for AdapterError {
    fn from(value: SignerError) -> Self {
        Self(value)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test panics on setup failure")]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_signer_round_trips_against_starknet_signers() {
        let key =
            Felt::from_hex("0x0139fe4d6f02e666e86a6f58e65060f115cd3c185bd9e98bd829636931458f79")
                .unwrap();
        let signer = LocalSigner::from_signing_key(key);
        let hash = Felt::from_hex("0x12345").unwrap();
        let [r, s] = signer.sign_hash(hash).await.unwrap();
        let pk = signer.public_key().await.unwrap();
        // Sanity: signature verifies against the public key.
        let vk = starknet_signers::VerifyingKey::from_scalar(pk);
        assert!(vk.verify(&hash, &Signature { r, s }).unwrap());
    }

    #[tokio::test]
    async fn simulation_signer_never_signs_and_is_interactive() {
        let signer = SimulationSigner::new(Felt::ONE);
        assert_eq!(signer.public_key().await.unwrap(), Felt::ONE);
        assert!(signer.is_interactive());
        assert!(matches!(
            signer.sign_hash(Felt::from_hex("0x12345").unwrap()).await,
            Err(SignerError::Config(_))
        ));
    }
}

# External Signers

Deadeye write paths can run without a private key in the CLI. This is useful
when the active profile is address-only and custody lives in a wallet companion,
hardware signer, multisig, or another agent-controlled signer.

## Universal Calldata Handoff

These commands support `--emit-calldata`:

```bash
deadeye trade execute <MARKET> --family normal --mean 3.30 --variance 0.1156 \
  --max-collateral 411 --emit-calldata

deadeye lp add <MARKET> --family normal --amount 100 --emit-calldata
deadeye lp remove <MARKET> --family normal --fraction 0.25 --emit-calldata
deadeye claim <MARKET> --family normal --emit-calldata
deadeye collateral claim-grant --emit-calldata
```

The CLI builds the exact account multicall it would submit, simulates it with
skip-validation, prints JSON, and exits without signing. The JSON includes:

- `account_address`
- `calls[]` with `contract_address`, `entry_point`, `entry_point_selector`, and
  hex `calldata`
- `preflight.is_valid`, `preflight.rejection`, and the simulation note
- trade collateral preflight when available

`trade execute --dry-run` is also keyless. Set `--address` or use an address-only
profile so simulation runs as the real account.

## Profile Configuration

External submission is selected per profile:

```toml
[profiles.strkd]
rpc_url = "https://example.invalid/rpc"
chain_id = "0x534e5f4d41494e" # SN_MAIN
address = "0x046cdf..."
signer = "external"

[profiles.strkd.external_signer]
kind = "strkd"
discovery = "port_lock"
token_path = "~/.config/deadeye/strkd.json"
allowance_strategy = "skip-estimate"
```

`endpoint = "http://127.0.0.1:<port>"` can be used instead of port-lock
discovery. Environment overrides:

- `DEADEYE_SIGNER=external`
- `DEADEYE_EXTERNAL_SIGNER_ENDPOINT=http://127.0.0.1:<port>`
- `DEADEYE_STRKD_PORT_LOCK=/path/to/port.lock`
- `DEADEYE_STRKD_TOKEN=<bearer-token>`
- `DEADEYE_STRKD_CLIENT_ID=<client-id>`

## strkd Adapter

The built-in adapter calls `wallet_addInvokeTransaction` on the strkd loopback
JSON-RPC endpoint.

Every POST uses:

- `Authorization: Bearer <token>`
- `X-Companion-Client: <client-id>`

Do not use `X-Companion-Token`; strkd may let unauthenticated status calls work
while authenticated wallet calls fail with `NOT_REGISTERED`.

Deadeye passes `chainId` on each `wallet_addInvokeTransaction` request. Avoid
`wallet_switchStarknetChain` in agents because it mutates shared wallet state.

By default the adapter sends `submit: true`, so strkd broadcasts through its
configured node and Deadeye reports the returned transaction hash when present.
If strkd is configured to sign only, the CLI reports that a signed transaction
was returned but no broadcast hash was available.

## Allowance And Estimation

Deadeye usually submits trade and LP operations as atomic multicalls, for
example `[approve, execute_trade]`. Some wallet companions estimate fees against
pre-transaction allowance and do not account for the leading in-transaction
`approve`; that can produce `INSUFFICIENT_ALLOWANCE` during estimation even
though the atomic multicall would execute.

The strkd adapter deliberately defaults to:

```toml
allowance_strategy = "skip-estimate"
```

This includes explicit resource bounds in the wallet request so the wallet does
not need to estimate through the stale allowance state. The bounds are generous;
unused gas is refunded by Starknet.

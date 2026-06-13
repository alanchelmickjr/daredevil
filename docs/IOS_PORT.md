# iPhone 13+ port

> iPhone is **target #1** for the portable runtime, not *the* target. The
> cross-platform strategy — one Rust core, thin platform shims, ONNX Runtime as
> the universal accelerator — lives in [`PORTABILITY.md`](PORTABILITY.md). This
> doc is the device-specific plan: what's hard on iOS, how each stage maps, and
> the phasing.

## The honest framing: re-implementation, not a code port

Daredevil today is Python (`torch` + `speechbrain` + `librosa` +
`pyroomacoustics` + `sounddevice` + `numpy/scipy`). None of that ships acceptably
on iOS — there is no App-Store-viable CPython runtime for real-time audio + ML.
So the iPhone "port" is: **the shared Rust core ([`PORTABILITY.md`](PORTABILITY.md))
behind a thin Swift shim.** The portable artifacts that survive verbatim are the
**awareness-map JSON schema** and the **slot contract** from
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## The constraint that shapes everything: the mics

- iPhone 13 has 3 physical mics, but **iOS does not expose raw simultaneous
  N-channel access to the built-in array.** You cannot feed stock iPhone mics into
  SRP-PHAT. The most you get is **2-channel "stereo"** capture (iPhone XS+, via
  `AVAudioSession` data-source orientation / polar patterns) — enough for a crude
  left/right azimuth, not 3D DOA.
- This maps **cleanly onto the existing degradation story**
  ([`BUILD_SPEC.md`](BUILD_SPEC.md), single-mic fallback): single/stereo mic →
  **WHO ✅ · WHAT ✅ · HOW ✅ · WHERE ❌.** Stage 1 spatial is the premium, and on a
  bare iPhone it stays the premium. On-thesis, not a compromise.
- **Full WHERE returns via the hardware module.** It presents as USB-C **UAC2**,
  and iOS supports class-compliant multichannel UAC2 input through `AVAudioEngine`
  (Lightning on 13/14 via the camera adapter; native USB-C on 15+). The exact
  software/hardware split already in the pitch survives the port:
  **phone = WHO/WHAT/HOW, module = unlocks WHERE.**

> **Validate early:** UAC2 multichannel capture on a real device, and the
> ECAPA→CoreML conversion. These two spikes de-risk the whole effort.

---

## Stage-by-stage mapping

| Daredevil piece | iOS approach |
|---|---|
| **Capture** (`sounddevice`) | `AVAudioEngine` tap on `inputNode`; `AVAudioSession` `.record`. Single/stereo built-in, or UAC2 module for multichannel. |
| **VAD** (Silero) | Ships as ONNX/CoreML, tiny — runs in the core as-is. Energy gate stays the cheap pre-filter. |
| **WHO — ECAPA-TDNN** | torch → ONNX → CoreML (`coremltools`) or `ort` + CoreML EP, inside the Rust core. ~22M params, runs on ANE. The headline and most feasible slice. |
| **WHAT — events** | **YAMNet, not PANNs.** ~4 MB, TFLite/CoreML-friendly. PANNs (300 MB) is a mobile non-starter — and [`MODELS.md`](MODELS.md) already names YAMNet the edge default. |
| **HOW — prosody** (librosa) | No model — F0 (YIN/pYIN), jitter, shimmer, HNR in the Rust core (`realfft`); the Swift shim does nothing here. |
| **Stage 3 router / tracker / SPRT / NMF** | Pure numeric → already in the shared Rust core. Smallest-risk part. |
| **Parallel slots** | Core uses real threads + ONNX dispatch to ANE, so "latency = max(slot)" holds — and the GIL caveat from the Python path disappears entirely. |
| **Identity store + crypto** (`cryptography`) | Shim provides **CryptoKit + Secure Enclave**; private key in **Keychain**. Voiceprints encrypted with a Secure-Enclave-backed key — a *stronger* privacy posture than desktop. |
| **Fleet sync** (Gun, JS/Node) | Two paths: (a) host Gun's SEA in **JavaScriptCore** (built into iOS) for fast parity; (b) native `URLSession` WebSocket + CryptoKit (SEA ≈ ECDSA P-256 + AES-GCM, ~1:1). Prototype (a), ship (b). |
| **Viz HUD** (web) | Native **SwiftUI** radar (better UX), or drop the existing web HUD in a `WKWebView` to start. |
| **MCP server** (stdio) | Doesn't fit the iOS app model. Reframe: expose the awareness map via **App Intents** / a loopback HTTP endpoint, or pipe it into an on-device LLM. |

---

## The on-device LLM loop — an iPhone-13-specific gotcha

The `<3 s` end-to-end (STT → small LLM) needs care on this device tier:

- **STT**: Apple's **Speech framework** with `requiresOnDeviceRecognition = true`
  is zero-dependency and free. whisper.cpp (Metal) is the heavier fallback.
- **LLM**: ⚠️ **Apple's Foundation Models (on-device LLM) require A17 Pro /
  iPhone 15 Pro+ — iPhone 13 cannot run Apple Intelligence.** On 13/14, use
  **MLX-Swift** or **llama.cpp** with a small Gemma/Phi, *or* — cleaner — have the
  phone emit only the awareness map and let a paired device (or the user's own Mac
  on LAN) run the LLM.
- **Thesis boundary:** streaming a child's voice to the **cloud** is exactly what
  COPPA forbids and what the whole pitch rejects (see [`PRIVACY.md`](PRIVACY.md)),
  so "thin client → cloud backend" is **off the table by design.** On-device, or
  on-trusted-LAN-device, only.

---

## Phasing (device level)

- [ ] **Phase 0 — Parity harness** (shared, see [`PORTABILITY.md`](PORTABILITY.md)):
      golden vectors from the synthetic scenes.
- [ ] **Phase 1 — WHO-only iPhone app.** VAD + ECAPA (CoreML) + 3-second
      enrollment + cosine match + Keychain/Secure-Enclave store. Single mic.
      The headline ("WHO comes first") and the smallest shippable, demo-able slice.
- [ ] **Phase 2 — Full single-mic map.** Add YAMNet (WHAT) + DSP prosody (HOW) +
      router with `SAFETY_CRITICAL` / `DISTRESS` overrides. Complete awareness map
      on the phone, minus WHERE.
- [ ] **Phase 3 — Fleet P2P.** CryptoKit sync across the user's own devices.
- [ ] **Phase 4 — WHERE via the module.** UAC2 multichannel → SRP-PHAT +
      beamforming in the core. The premium unlock.
- [ ] **Phase 5 — Local LLM loop.** Speech framework + MLX/llama.cpp.

## Risks to track

1. **Built-in mics can't do real DOA** — WHERE depends on the module (or accept
   stereo-only bearing). Spike UAC2 capture early.
2. **Always-on listening is a product/policy problem** as much as a technical one —
   iOS background-audio mode, the permanent orange mic indicator, battery/thermal,
   and strict App Store review of always-listening apps. VAD-gating (already in the
   design) is the battery answer; explicit consent UX is the review answer.
3. **Model-conversion numeric drift** after CoreML int8 — caught by the Phase 0
   golden vectors.
4. **iPhone 13 ≠ Apple Intelligence** — don't design the LLM loop around Apple's
   on-device model on this tier.

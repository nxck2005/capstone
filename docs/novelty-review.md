# Novelty review — 2026-10-01

A stress test of the project's novelty claim against published work, written to answer one viva
question: *why is this project novel?* Not normative — `spec/SPEC.md` governs, and DEC-13 is still
the recorded novelty position until an AM entry changes it.

Scope: about 25 web searches and five full papers read. This is not a systematic review, so every
"first" below must be said as "to our knowledge". Lokumarambage et al. 2026 was paywalled; only its
abstract was checked.

## The short answer

> To our knowledge, we're the first to measure how much of deep JSCC's advantage in wireless image
> classification comes from the way it is compared rather than from the learning itself: once the 5G
> digital link is tuned for each signal level and its receiver classifier is trained on compressed
> images, the advantage shrinks to the signal levels where the digital link fails.

Plain version: *other papers say the AI system beats normal digital transmission; we show most of
that win disappears when the comparison is made fair, and we measure which unfair choices create it.*

Shorter: *we test what end-to-end learning actually buys, not just whether it wins.*

## What other people have already done

| Prior work | What it did | What it means for us |
|---|---|---|
| Huang et al., IEEE IoT-J 2024 ([arXiv 2302.13580](https://arxiv.org/pdf/2302.13580)) — **not yet cited in the paper** | Sent learned features digitally over LDPC, next to BPG+LDPC and deep JSCC, for classification. Saw the LDPC cliff; the learned digital system beat DJSCC once its packets arrived. | We cannot claim to be first to show "learned features sent digitally match DJSCC until the cliff". Theirs used one fixed code rate, a reconstruction-only DJSCC, and a classifier trained on clean images. Must be cited. |
| Jankowski et al., JSAC 2021 ([arXiv 2007.10915](https://arxiv.org/abs/2007.10915)) | Task-aware digital vs task-aware analog JSCC for image retrieval; JSCC won clearly. | Their digital side **assumed perfect (capacity-achieving) channel codes**. We use a real LDPC chain, and our result points the other way. |
| Lokumarambage et al. 2026 ([OuluREPO](https://oulurepo.oulu.fi/handle/10024/64480)) | VQ semantic system vs BPG vs JSCC with LDPC/polar codes, including classification. | "First three-family comparison" is taken. |
| SwinJSCC, 2025 | BPG + 5G LDPC with the setting chosen per SNR. | Reconstruction only. A per-SNR 5G baseline is not new. |
| Ren et al. 2025 ([arXiv 2501.04285](https://arxiv.org/abs/2501.04285)); D²-JSCC ([arXiv 2403.07338](https://arxiv.org/html/2403.07338)) | Separate coding can match or beat JSCC; avoiding the cliff is described as *the* JSCC benefit. | "DJSCC wins by avoiding the cliff" is common knowledge (DEC-13 already says so). |
| Janeiro et al. 2023 ([arXiv 2304.04518](https://arxiv.org/abs/2304.04518)) | Retraining vision models on compressed images recovers about 82% of the accuracy lost to compression. | "Training on compressed images helps" is known in computer vision. Our claim must be about its effect on learned-vs-digital comparisons. |
| Typical DJSCC-vs-digital classification papers ([2512.19764](https://arxiv.org/html/2512.19764), [2607.28907](https://arxiv.org/html/2607.28907), Huang 2024) | Digital outputs scored with a classifier trained on clean images, or no digital baseline at all. | **This is the gap.** No paper found controls this or measures how much it moves the crossover. |

## The three candidate claims

**A. "A controlled three-way comparison on one shared 5G chain."** True but incremental, and it
sounds like a communications project. Keep it as the *method*, not the headline.

**B. "DJSCC's advantage is avoiding outage, not a better representation."**
- Not new as an idea (Huang 2024; the cliff story is standard).
- Holds only at bandwidth ratio 1/6. At 1/24 DJSCC also wins from −1 to +3 dB, where packets do arrive.
- ER-9 and DJSCC differ in encoder and head, which the paper already admits.
- "Features 82%, images 89%" compares a single linear layer with ResNet-18 — classifier size, not
  representation. Do not claim the representation does not help.

**C. "The receiver's classifier decides who wins."** The strongest claim:
- Large: up to 43 points; moves the crossover about 3 dB at 1/6 and about 6 dB at 1/24.
- Already measured, and too large to vanish on the test split.
- An ML problem (distribution shift at the receiver), not a channel-coding one.
- Not controlled in any learned-vs-digital comparison found.

## A weakness to decide on before G-12

The digital systems' lowest LDPC code rate is 1/3 (`params.baseline.ldpc_rates`). That floor is the
only reason nothing decodes below −5 dB at 1/6. Real 5G NR goes much lower (TS 38.214 MCS index 0 is
QPSK at 120/1024; the low-spectral-efficiency table reaches 30/1024), and the spec already allows
BG2 down to rate 0.2.

Shannon limits at k = 12,800 (theory only, nothing simulated):

| Setting | Lowest workable Es/N0 (theory) | Bits carried | ER-9 width that fits → measured accuracy |
|---|---|---|---|
| BPSK, rate 1/3 (current floor) | −5.3 dB (measured cliff ≈ −4.5) | ~4,270 | D=2048 → 82.0% |
| BPSK, rate 1/5 (BG2 minimum) | −8.0 dB | ~2,560 | D=1024 → 81.8% |
| QPSK, rate 30/1024 (NR low-SE) | −13.8 dB | ~750 | D=256 → 75.9% |

If the lower rates sit about 1 dB above their limit as rate 1/3 does, a digital link using
NR-realistic rates would deliver about 76–82% where DJSCC scores about 73–76%. At 1/6 that could
remove DJSCC's only winning region and most of H1's footing. The paper already makes this argument
for the label control ("a link built for a 4-bit payload could use a much lower code rate").

Options (owner's decision):
1. **Disclose** it in Limitations and qualify every "DJSCC is the only system below −5 dB" claim with
   "within a rate set whose lowest rate is 1/3".
2. **Add lower rates before the freeze manifest.** DEC-16 allows changes that strengthen the
   baseline, and test is unread. Cost: new BLER curves, re-selection, a validation re-run.

Either way the short answer above still stands: a stronger baseline shrinks DJSCC's advantage
further, which is the same direction the claim already points.

## Prepared answers

- *"So what exactly is new?"* — Two controls: a receiver classifier trained on compressed images,
  which alone moves the crossover by 3–6 dB, and learned features sent over the identical 5G link,
  which shows the remaining gain is not a better representation.
- *"Huang / Lokumarambage already did this."* — They compared the families. Huang used one fixed
  code rate and a clean-trained classifier; Jankowski assumed perfect codes; none controlled the
  receiver classifier or tuned the link per SNR with a learned-feature control on the same chain.
- *"There's no new architecture."* — Several highly cited ML papers propose none and instead show
  that claimed gains shrink under fair comparison: Musgrave et al., *A Metric Learning Reality
  Check* (ECCV 2020); Ferrari Dacrema et al., *Are We Really Making Much Progress?* (RecSys 2019);
  Melis et al., *On the State of the Art of Evaluation in Neural Language Models* (ICLR 2018); Lucic
  et al., *Are GANs Created Equal?* (NeurIPS 2018). The variables tested here are ML ones:
  end-to-end vs modular learning, receiver distribution shift, and training-distribution
  generalisation.

What the short answer deliberately avoids: saying which system is better (the DJSCC model is a
small 1.5M-parameter CNN that tops out near 84%), and saying "mostly outage" (true only at 1/6 and
only with the rate-1/3 floor).

## Paper changes needed for the claim to hold

1. Cite Huang 2024 in Related Work; note that Jankowski's digital arm assumed perfect codes.
2. Cite Janeiro 2023 beside the classifier result, framed as its effect on learned-vs-digital
   comparisons.
3. Put the classifier finding in the abstract and a contributions list; replace the defensive
   "not a new communication family" sentence in the Introduction.
4. Retitle around the question, e.g. *How Much of Deep JSCC's Advantage Survives a Fair Comparison?*
5. Add the rate-1/3 floor to Limitations and qualify the Discussion and Conclusion, or amend the
   rate set before the freeze.
6. Drop "the representation doesn't help"; fix the single-seed 2.6 pp figure and the analytic H4 MDE
   (both already listed in `NEXT.md`).
7. If adopted, record the new novelty position as an AM entry in `SPEC.md` §17 replacing DEC-13's
   lead. DEC-13's bit-accounting claim is still unverified under its own PR-1 condition.

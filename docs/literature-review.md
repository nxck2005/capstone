# Literature Review: Task-Oriented Deep Joint Source–Channel Coding

**Review state:** First Review baseline, 2026-08-11  
**Project scope:** image classification over a bandwidth-constrained AWGN link  
**Normative source:** [`spec/SPEC.md`](../spec/SPEC.md)

## 1. Review question and method

This review asks a narrow engineering question:

> At short blocklengths and a fixed number of complex channel uses, can an end-to-end learned, task-aware joint source–channel code preserve image-classification accuracy more effectively than a properly tuned JPEG 2000 + 5G NR LDPC transmission chain?

The review covers four bodies of work: information-theoretic limits; learned image compression; neural joint source–channel coding (JSCC); and task-oriented semantic communication. Sources were selected for direct relevance to finite-blocklength transmission, image coding, channel adaptation, or downstream inference. Standards and implementation manuals are separated into the companion [standards and tools register](standards-and-tools-register.md). This is a structured narrative review, not a systematic review or meta-analysis.

The project deliberately makes a narrower claim than much of the semantic-communications literature. It does not claim that conventional digital systems cannot carry semantics, nor that learned transmission is universally superior. It tests one preregistered image-classification protocol under identical SNR, sample identity, noise identity, and complex-symbol budget. The task-aware digital control is necessary because a digital system can transmit learned features rather than reconstructed pixels.

## 2. Foundations: separation, finite blocklength, and distortion

Shannon's source–channel separation result establishes the asymptotic optimality of independently designed source and channel codes under its assumptions [1]. It does not say that a practical short-packet concatenation is optimal at finite latency. Gastpar, Rimoldi, and Vetterli show that uncoded or joint mappings can be optimal when the source, distortion, channel, and input-cost functions are appropriately matched [2]. These results motivate joint design but do not predict that a neural implementation will win in this experiment.

Finite-blocklength theory makes the practical qualification explicit. Polyanskiy, Poor, and Verdú characterize the rate back-off from channel capacity at finite blocklength through channel dispersion [3]. Kostina and Verdú extend non-asymptotic analysis to lossy joint source–channel coding and quantify the interaction between source and channel dispersion [4]. The implication for this project is structural: an image packet with a fixed, modest channel-use budget can incur source-coding overhead, CRC and segmentation overhead, rate-matching loss, and a nonzero block-error probability even when asymptotic separation is sound. These costs must be measured rather than assumed.

The relevant output distortion is also unusual. Classical rate–distortion work optimizes a reconstruction metric. This project ultimately scores top-1 task accuracy, while retaining PSNR and SSIM as secondary measures. Blau and Michaeli's perception–distortion result warns that pixel distortion and perceptual realism are not interchangeable [17]. Task accuracy is a third objective again: an image can be visually plausible yet alter the decision-relevant feature, or look poor while preserving the class. This motivates explicit task-aware training and reporting rather than treating PSNR as a proxy for classification.

## 3. Learned image compression

Modern learned image compression established that analysis/synthesis transforms and entropy models can be optimized end to end for rate–distortion performance. Ballé, Laparra, and Simoncelli use nonlinear transforms and a differentiable quantization surrogate [11]. Ballé et al. add a learned scale hyperprior to model spatial dependencies [12]. Minnen, Ballé, and Toderici combine hierarchical and autoregressive priors [13]. Theis et al. develop compressive autoencoders [14], while Toderici et al. demonstrate variable-rate recurrent compression [15]. Cheng et al. combine discretized Gaussian-mixture likelihoods with attention [16]. These works materially improve the source-coding side of a separated link.

They do not, by themselves, solve this project's question. Their primary objective is reconstruction rate–distortion, and a bitstream still needs channel protection. A learned source codec could strengthen a digital baseline, but it would add another trained model and another corpus-dependent comparison. The preregistered baseline therefore uses JPEG 2000, a mature source codec with low-rate scalability and a raw-codestream mode, and treats learned source compression as related work rather than silently changing the comparator. The baseline is not held fixed across SNR: codec quality, LDPC rate, and modulation are selected per SNR on validation data.

## 4. Neural joint source–channel coding for images

Bourtsoulatze, Kurka, and Gündüz introduced the modern DeepJSCC image-transmission architecture: convolutional encoder and decoder networks connected by a non-trainable channel layer [5]. Their AWGN and fading results established the characteristic comparison now called graceful degradation: reconstruction quality degrades continuously as channel conditions move away from those seen in training, whereas a separated digital chain can suffer a decoding cliff.

Subsequent work expands the operating assumptions:

- DeepJSCC-f exploits channel-output feedback and demonstrates that learned joint coding can use information ignored by a conventional separated design [6]. Feedback is outside this project's Tier 1 scope.
- DeepJSCC-l supports successive refinement and multiple descriptions, addressing variable available bandwidth [7]. This project's progressive packetisation sensitivity is related, but its headline comparison keeps the preregistered symbol budget fixed.
- Model-driven DeepJSCC with differentiable OFDM blocks addresses multipath fading and nonlinear clipping [8]. This project uses abstract baseband AWGN in Tier 1 and registers OFDM/resource mapping as excluded from its standards claim.
- WITT replaces the CNN backbone with a Swin Transformer and adapts latent features to channel state [9]. SwinJSCC adds joint adaptation to channel condition and target rate [10]. These papers show that backbone capacity and adaptation policy can materially affect neural JSCC results; the present project therefore freezes its smaller residual CNN before headline training and does not generalize results to transformer systems.
- DeepJSCC-Q constrains the learned transmitter to a finite channel-input alphabet [24]. It is especially relevant to the project's attribution control: hardware-compatible or digital symbol constraints do not make task-aware transmission impossible.

Most image-JSCC studies optimize a reconstruction loss and report PSNR, SSIM, or perceptual measures. This project instead uses a dual-head decoder and the objective `CE + λ × MSE`, preserving a reconstruction head while optimizing the downstream classification decision. That changes the scientific question: a gain may arise from task-aware representation, joint source–channel coding, or both.

## 5. Task-oriented and semantic communication

Task-oriented communication optimizes information delivery for a receiver-side task rather than for faithful reproduction of every source sample. Jankowski, Gündüz, and Mikolajczyk compare digital feature compression and analog JSCC for wireless image retrieval [18]. Their inclusion of both task-aware digital and analog approaches is directly relevant: a fair attribution study must not compare a task-aware learned system only against a task-agnostic reconstruction pipeline.

Shao, Mao, and Zhang formulate task-oriented edge inference using a variational information bottleneck and variable-length feature encoding [19]. Xie et al.'s DeepSC uses a Transformer and semantic similarity for text transmission [20]; their later multi-user work targets image retrieval, machine translation, and visual question answering [21]. These systems establish that the task and metric must be explicit: sentence similarity, retrieval rank, VQA accuracy, and image-classification accuracy are different contracts.

Kang et al. select image regions for UAV scene classification under latency constraints [22]. Liu et al. jointly address semantic task distortion and image reconstruction through a rate–distortion formulation [23]. Together with DeepJSCC-Q [24], this literature shows a continuum rather than a binary split between “semantic” and “digital”: systems can transmit pixels, compressed reconstructions, discrete learned features, or continuous task-aware latents.

The project's ER-9 control occupies the missing middle. It quantizes learned features and carries them through the same LDPC and modulation chain at matched complex-symbol budget. Comparing it with the reconstruction baseline and continuous DJSCC gives three distinct systems:

1. **Task-agnostic separated:** JPEG 2000 → LDPC → modulation → reconstruction → frozen classifier.
2. **Task-aware digital:** learned features → quantization → the same digital physical layer → classifier.
3. **Task-aware joint:** encoder → differentiable AWGN → dual-head decoder/classifier.

Only this three-way comparison can distinguish a task-aware representation benefit from a joint-coding benefit within the limits of the experiment.

## 6. The classical baseline and fairness problem

Low-density parity-check codes approach capacity with practical iterative decoding [25, 26]. Modern coding texts explain the design and finite-length behavior of sparse-graph codes [27]. JPEG 2000 provides wavelet-based embedded coding and low-rate operation [28, 29]. These are credible components, but merely naming them does not produce a fair baseline.

A fair comparison requires all of the following:

- identical image identities and frozen train/validation/test partitions;
- identical complex channel-use budget `k` at each bandwidth ratio;
- one SNR definition, here $E_s/N_0$ per normalized complex channel use;
- the same keyed AWGN realization where paired comparison is possible;
- exact accounting for source-container, packet, CRC, filler, and rate-matching overhead;
- codec quality, LDPC rate, and modulation tuned on validation rather than selected to make the baseline weak;
- explicit outage behavior when decoding fails, instead of silently dropping failures;
- no test-set tuning; and
- paired uncertainty intervals for the final task-accuracy difference.

The modulation point is important. A baseline fixed to QPSK across the whole SNR range would be artificially constrained at high SNR. The preregistered baseline may choose BPSK, QPSK, or 16-QAM and one of four LDPC rates per SNR, subject to feasibility and a measured BLER table. The fixed-modulation curve remains as a labelled secondary reference, not the headline.

## 7. Synthesis of representative work

| Work | Source/task | Channel or link | Main objective | Relevance | Limitation for this study |
|---|---|---|---|---|---|
| Shannon [1] | General | Memoryless channel | Asymptotic reliable communication | Separation benchmark | Does not settle finite packet performance |
| Polyanskiy et al. [3] | Channel messages | General/AWGN examples | Finite-blocklength coding rate | Explains practical rate back-off | No image semantics |
| Kostina & Verdú [4] | Lossy source | Joint source/channel | Non-asymptotic excess distortion | Formal finite-length JSCC basis | Not a learned implementation |
| Ballé et al. [11, 12] | Images | Error-free bitstream | Rate–distortion | Strong source coding context | No noisy-channel/task objective |
| Bourtsoulatze et al. [5] | Images | AWGN/fading | Reconstruction DeepJSCC | Core architecture and graceful degradation | Reconstruction, not classification |
| Kurka & Gündüz [6, 7] | Images | Feedback / variable bandwidth | Reconstruction and refinement | Adaptation and practical JSCC extensions | Different link assumptions |
| Yang et al. [8] | Images | OFDM multipath | Model-driven reconstruction JSCC | Hardware-facing waveform structure | OFDM outside Tier 1 claim |
| WITT/SwinJSCC [9, 10] | Images | AWGN/fading | Transformer reconstruction JSCC | Shows backbone/rate adaptation effects | Larger architecture, reconstruction metric |
| Jankowski et al. [18] | Image retrieval | Static/fading wireless | Retrieval accuracy | Digital and analog task-aware comparison | Retrieval rather than classification |
| Shao et al. [19] | Edge inference | Dynamic channels | Information bottleneck/task accuracy | Learned feature transmission | Different feature and latency protocol |
| DeepSC family [20, 21] | Text/retrieval/VQA | Wireless channel models | Semantic task metrics | Demonstrates task-specific contracts | Not directly comparable to image classification |
| Kang et al. [22] | UAV scenes | Variable channel | Classification/latency | Direct classification precedent | Region-selection/DRL design differs |
| Liu et al. [23] | Images/multiple tasks | Rate-constrained link | Reconstruction + semantic distortion | Closest dual-objective motivation | Different rate formulation |
| DeepJSCC-Q [24] | Images | Finite constellation | Reconstruction JSCC | Bridges continuous and digital inputs | Does not perform this attribution protocol |

## 8. Identified gap and project contribution

The literature separately establishes strong learned source coding, graceful neural JSCC, and task-aware feature transmission. It does not eliminate the need for a controlled, reproducible comparison in which:

- the classical arm is a real JPEG 2000 + standards-derived 5G NR LDPC pipeline;
- the baseline is tuned per SNR and may adapt its modulation;
- every arm receives exactly the same channel-use budget;
- failures remain in the denominator through a preregistered outage policy;
- a task-aware digital control separates representation learning from joint coding; and
- final claims use paired image-level outcomes and confidence intervals.

That protocol is the project's main scientific contribution. The implementation contribution is a lineage-bound experimental system in which dataset bytes, manifests, random identities, code paths, contracts, and evidence artifacts are authenticated before the test split is opened. Neither contribution guarantees a positive result. Completion means executing the protocol without changing it in response to observed results.

The primary preregistered hypothesis is a positive paired top-1 accuracy difference at three consecutive low-SNR points. A curve crossing is reported if observed but is not required. The strongest defensible conclusion will be conditional: for the tested dataset, architecture, symbol budgets, AWGN definition, baseline search space, and seed protocol.

## 9. Review implications for design

1. **Use task accuracy as the primary endpoint.** Reconstruction metrics remain diagnostic, not substitutes.
2. **Preserve a reconstruction head.** It makes semantic degradation inspectable and supports λ calibration rather than optimizing an opaque classifier alone.
3. **Do not handicap separation.** Tune codec, LDPC rate, and modulation on validation; account for all bytes and all failures.
4. **Keep ER-9.** Without the task-aware digital control, a learned-vs-classical gap cannot be attributed to JSCC.
5. **Freeze before test.** Architecture, ratios, λ, operating points, checkpoints, and analysis code must be fixed before the single test campaign.
6. **Report scope.** AWGN Tier 1 results do not establish performance under fading, synchronization error, RF nonlinearity, or real SDR timing.

## References

1. C. E. Shannon, “A Mathematical Theory of Communication,” *Bell System Technical Journal*, 1948. <https://doi.org/10.1002/j.1538-7305.1948.tb01338.x>
2. M. Gastpar, B. Rimoldi, and M. Vetterli, “To Code, or Not to Code: Lossy Source–Channel Communication Revisited,” *IEEE Transactions on Information Theory*, 2003. <https://infoscience.epfl.ch/entities/publication/69ffb6df-cf05-4921-ae06-d6eea7332313>
3. Y. Polyanskiy, H. V. Poor, and S. Verdú, “Channel Coding Rate in the Finite Blocklength Regime,” *IEEE Transactions on Information Theory*, 2010. <https://ieeexplore.ieee.org/document/5452208/>
4. V. Kostina and S. Verdú, “Lossy Joint Source–Channel Coding in the Finite Blocklength Regime,” *IEEE Transactions on Information Theory*, 2013. <https://arxiv.org/abs/1209.1317>
5. E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, “Deep Joint Source–Channel Coding for Wireless Image Transmission,” *IEEE Transactions on Cognitive Communications and Networking*, 2019. <https://arxiv.org/abs/1809.01733>
6. D. B. Kurka and D. Gündüz, “DeepJSCC-f: Deep Joint Source–Channel Coding of Images with Feedback,” *IEEE Journal on Selected Areas in Information Theory*, 2020. <https://arxiv.org/abs/1911.11174>
7. D. B. Kurka and D. Gündüz, “Bandwidth-Agile Image Transmission with Deep Joint Source–Channel Coding,” *IEEE Transactions on Wireless Communications*, 2021. <https://arxiv.org/abs/2009.12480>
8. M. Yang, C. Bian, and H.-S. Kim, “Deep Joint Source Channel Coding for Wireless Image Transmission with OFDM,” 2021. <https://arxiv.org/abs/2101.03909>
9. K. Yang, S. Wang, J. Dai, K. Tan, K. Niu, and P. Zhang, “WITT: A Wireless Image Transmission Transformer for Semantic Communications,” *ICASSP*, 2023. <https://arxiv.org/abs/2211.00937>
10. K. Yang, S. Wang, J. Dai, X. Qin, K. Niu, and P. Zhang, “SwinJSCC: Taming Swin Transformer for Deep Joint Source–Channel Coding,” 2023. <https://arxiv.org/abs/2308.09361>
11. J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-End Optimized Image Compression,” *ICLR*, 2017. <https://arxiv.org/abs/1611.01704>
12. J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational Image Compression with a Scale Hyperprior,” *ICLR*, 2018. <https://arxiv.org/abs/1802.01436>
13. D. Minnen, J. Ballé, and G. Toderici, “Joint Autoregressive and Hierarchical Priors for Learned Image Compression,” *NeurIPS*, 2018. <https://arxiv.org/abs/1809.02736>
14. L. Theis, W. Shi, A. Cunningham, and F. Huszár, “Lossy Image Compression with Compressive Autoencoders,” *ICLR*, 2017. <https://arxiv.org/abs/1703.00395>
15. G. Toderici et al., “Full Resolution Image Compression with Recurrent Neural Networks,” *CVPR*, 2017. <https://arxiv.org/abs/1608.05148>
16. Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules,” *CVPR*, 2020. <https://arxiv.org/abs/2001.01568>
17. Y. Blau and T. Michaeli, “The Perception–Distortion Tradeoff,” *CVPR*, 2018. <https://arxiv.org/abs/1711.06077>
18. M. Jankowski, D. Gündüz, and K. Mikolajczyk, “Wireless Image Retrieval at the Edge,” *IEEE Journal on Selected Areas in Communications*, 2021. <https://arxiv.org/abs/2007.10915>
19. J. Shao, Y. Mao, and J. Zhang, “Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach,” 2021. <https://arxiv.org/abs/2102.04170>
20. H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang, “Deep Learning Enabled Semantic Communication Systems,” *IEEE Transactions on Signal Processing*, 2021. <https://arxiv.org/abs/2006.10685>
21. H. Xie, Z. Qin, X. Tao, and K. B. Letaief, “Task-Oriented Multi-User Semantic Communications,” 2021. <https://arxiv.org/abs/2112.10255>
22. X. Kang, B. Song, J. Guo, Z. Qin, and F. R. Yu, “Task-Oriented Image Transmission for Scene Classification in Unmanned Aerial Systems,” 2021. <https://arxiv.org/abs/2112.10948>
23. F. Liu, W. Tong, Y. Yang, Z. Sun, and C. Guo, “Task-Oriented Image Semantic Communication Based on Rate–Distortion Theory,” 2022. <https://arxiv.org/abs/2201.10929>
24. T.-Y. Tung, D. B. Kurka, M. Jankowski, and D. Gündüz, “DeepJSCC-Q: Channel Input Constrained Deep Joint Source–Channel Coding,” 2021. <https://arxiv.org/abs/2111.13042>
25. R. G. Gallager, “Low-Density Parity-Check Codes,” *IRE Transactions on Information Theory*, 1962. <https://doi.org/10.1109/TIT.1962.1057683>
26. D. J. C. MacKay, “Good Error-Correcting Codes Based on Very Sparse Matrices,” *IEEE Transactions on Information Theory*, 1999. <https://doi.org/10.1109/18.748992>
27. T. Richardson and R. Urbanke, *Modern Coding Theory*, Cambridge University Press, 2008. <https://doi.org/10.1017/CBO9780511791338>
28. D. S. Taubman and M. W. Marcellin, *JPEG2000: Image Compression Fundamentals, Standards and Practice*, Springer, 2002. <https://doi.org/10.1007/978-1-4615-0799-4>
29. A. Skodras, C. Christopoulos, and T. Ebrahimi, “The JPEG 2000 Still Image Compression Standard,” *IEEE Signal Processing Magazine*, 2001. <https://doi.org/10.1109/79.952804>
30. T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006. <https://doi.org/10.1002/047174882X>

# Plain-language summary

Climate models can't directly simulate atmospheric gravity waves — ripples in
the air, often triggered by mountains and storms, that carry momentum high
into the atmosphere and shape global wind patterns. Modelers therefore use
simplified "parameterizations" for them, and recently, neural networks
trained on high-resolution data have started replacing these hand-built
schemes. One influential line of work found that giving the network a wider
horizontal view (seeing weather *around* a location, not just above it) makes
predictions much better — and it is natural to hope this is because the
network learned real wave physics, like how waves travel sideways.

We opened these networks up — the actual published models — and tested that
hope directly, the way one would test a scientific claim: with pre-registered
hypotheses (30 of them), controls, and interventions on the models'
internals. Most hypotheses died honestly. What survived paints an unexpected
picture:

1. **The network cheats geography.** The best model was never told where on
   Earth it is looking, yet it behaves as if it knows. It infers position
   from a technical artifact (how convolutions treat the map's edges) and
   uses it as a hidden "you are here" signal. About 60% of its most
   celebrated advantage — predicting realistic variability in wave
   strength — comes from this memorized geographic prior: it has learned
   *that* the Andes make strong waves, and injects the right variability at
   that map location. Shift the input map sideways (which changes no
   physics) and this skill collapses.

2. **The wide view helps, but not the way physics would.** The network
   genuinely uses information from ~1500 km around a point — but uniformly
   in all directions, unrelated to how waves actually propagate, and mostly
   in stormy, windy regions. It reads the weather *regime*, not the wave
   paths.

3. **The physics isn't in there.** We built a lie-detector test: modify one
   physical aspect of a real weather profile (e.g., insert a wind reversal
   that should physically block waves) and watch the response. On a small
   testbed model that provably learned its physics, the test fires clearly.
   On the climate models, it stays silent: no blocking response, wrong drag
   directions, inconsistent scaling. Their skill comes from pattern-matching
   weather regimes, not from wave physics.

Why it matters: these networks perform well on the data they were trained
for, and nothing here says otherwise. But climate projection asks models to
work in conditions nobody has observed yet. A network that succeeded by
memorizing geography and regime statistics gives little reason for such
trust — and our tests give the community a concrete way to check. We also
found practical landmines (a public dataset silently incompatible with the
public models trained on it; a skill-metric that can be gamed by inflating
predictions) that anyone building on these resources should know about.

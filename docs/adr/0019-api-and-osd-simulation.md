# ADR-0019: Simulating aspirin API synthesis and tablet manufacture

Status: Accepted
Date: 2026-09-26
Refines: ADR-0008 (the simulator as data source)

## Context

ADR-0018 adds an API site (Tuas) and an oral-solid-dose site (Freiburg). ADR-0008's
reasons for a simulator apply to both: known ground truth, injectable faults and
levers with real optima. The bioreactor model is mechanistic enough that the AI has
something real to learn. The new processes need the same, or the AI results on them
mean nothing.

## Decision

**Aspirin API at Tuas.** Batch of 500 kg salicylic acid (SA), acetylated with acetic
anhydride (Ac2O) in acetic acid, cooled to crystallise, then filtered, washed and
dried. About 30 hours on the train.

| Unit | Operations | Phases |
| --- | --- | --- |
| RX-201 reactor-crystallizer | Charge → Reaction → Crystallization → Transfer | TEMP_CTRL, AGIT_CTRL, DOSE_ADD (monitors temperature: the exotherm) |
| FD-202 filter-dryer | Filtration → Washing → Drying → Discharge | FILTER_CTRL, TEMP_CTRL, VAC_CTRL (monitors moisture), AGIT_CTRL |

The mechanisms are:

- Second-order acetylation with Arrhenius kinetics and a side reaction to
  acetylsalicylsalicylic acid, which is faster when hot.
- A cooling crystallisation with a moment model. Fast cooling gives high
  supersaturation, many nuclei and fines.
- Fines blind the cake and pass the cloth.
- Hydrolysis back to SA while the cake is wet and warm.

In-line PAT (Raman conversion, FBRM chord length, NIR cake moisture) goes to `pv/*`.
LIMS publishes the reaction IPC and the CoA (assay, free SA, related substances, LOD,
D50, yield) to `lab/*`.

**Levers** (the recipe's PARs): reaction temperature 80–90 °C, Ac2O ratio 1.10–1.40
mol/mol, reaction hold 2–4 h, cooling rate 5–20 °C/h, dryer jacket 40–60 °C. Each has
an interior optimum for **API yield (%)** through competing mechanisms. For example,
more Ac2O converts more SA but leaves more acetic acid, so more product stays in the
mother liquor.

**Aspirin 500 mg tablets at Freiburg.** Aspirin hydrolyses in moisture, so the route
is dry granulation. A batch is 400,000 tablets from about 240 kg of blend (200 kg of
API plus MCC, maize starch and stearic acid; magnesium stearate is incompatible with
aspirin). About 12 hours on the train.

| Unit | Operations | Phases |
| --- | --- | --- |
| BL-301 bin blender | Charge → Blending → Lubrication → Discharge | BLEND_CTRL (monitors NIR blend RSD) |
| RC-302 roller compactor | Compaction | COMPACT_CTRL (monitors ribbon density), MILL_CTRL |
| TP-303 rotary tablet press | Compression | TABLET_CTRL (monitors the checkweigher), FEED_CTRL |

The mechanisms are:

- Blend uniformity that decays with revolutions, slower for coarse API.
- Ribbon density from the specific roll force, and granule size and fines from ribbon
  density.
- Tablet hardness from compression force, reduced by work-hardening at high roll
  force and by over-lubrication.
- Dissolution that is slower for hard tablets, coarse API and long lubrication.
- Weight variation from fines, turret speed and the feed-frame ratio.
- Capping at high force and speed.
- Sticking when under-lubricated.
- Free SA that grows with room humidity and exposure time.

The checkweigher rejects out-of-weight tablets.

**Levers:** lubrication time 2–6 min, specific roll force 4–10 kN/cm, main
compression force 10–20 kN, turret speed 30–60 rpm, feed-frame speed 20–40 rpm. Each
has an interior optimum for **tablet yield (%)** (good tablets over theoretical)
through rejects. Dissolution (Q ≥ 80% at 30 min), hardness, friability, content
uniformity (AV) and free SA (≤ 0.3%) are reported in the CoA.

**Faults** (labelled on `_sim/faults` as before; ADR-0012):

| Process | Fault | Mechanism | Observable |
| --- | --- | --- | --- |
| API | `jacket_fouling` | Reactor jacket heat transfer decays | Jacket runs colder while the reactor lags its cooling ramp |
| API | `dosing_meter_drift` | Ac2O flowmeter reads high, so less is dosed than shown | Dose total looks right; Raman conversion rises slower; smaller exotherm |
| API | `agitator_degradation` | Impeller slips on its shaft | Power draw falls at the same speed; slower reaction and heat transfer |
| API | `filter_blinding` | Cloth resistance grows | Filtrate flow collapses at the same N2 pressure |
| API | `vacuum_leak` | Dryer leak | Vacuum rises; moisture falls slowly; free SA rises |
| OSD | `roll_force_drift` | Force transducer reads high, so true force is lower | Force PV at SP; ribbon density and roll gap drift |
| OSD | `punch_sticking` | Product builds up on the punch faces | Ejection force climbs; rejects rise |
| OSD | `hopper_bridging` | Granules bridge in the press hopper | Force and weight scatter; RSD and rejects rise |
| OSD | `hvac_humidity` | Dehumidifier fails | Room RH climbs; free SA in the product rises |
| both | `stuck_sensor` | As for the bioreactor | Bit-identical values |

Every one has a harness test that the anomaly layer catches it (CLAUDE.md).

**Backfill history.** 140 batches per new process by default, each with its own
Latin-hypercube process-characterisation campaign (30% of batches), two recipe
versions (the older one further from the optimum) and about 15% faulty batches. The
bioreactor history is unchanged.

## Consequences

- The simulator gains two engines (`simulator/api/`, `simulator/osd/`). They share the
  bioreactor's message types, clock, sensor noise model and wire mapping, so the edge
  adapter, historian and backfill treat them the same.
- The models are plausible, not calibrated. Numbers such as yields, times and rates
  are in the right range for the demo and chosen so that faults and levers leave
  learnable signatures. They are not process knowledge anyone should reuse.
- `simulator.truth` gains a ground-truth outcome for train runs, so the optimizer can
  be evaluated on every process the same way (ADR-0015).

## Alternatives considered

- **Wet granulation.** It is the textbook OSD route and has more tags to show, but it
  is wrong for aspirin, which hydrolyses.
- **Empirical lookup tables instead of mechanisms.** They are cheaper, but faults
  would then have to be painted onto the outputs, and the PV-looks-fine-while-the-
  process-drifts faults (dosing meter, roll force) need a mechanism under the sensor.
- **Film coating and packaging.** More units, but nothing new for the UNS or the AI.
  They are a later increment if a customer asks.

Qubery
======

Controlling a simulated quantum computer with visually tracked origami cubes.

Notes
=====

**Arrow Values**

    Arrow direction determines phase, normalization determines magnitude
    · = 0
    → ∥  1
    ↗ ∥  1+i
    ↑ ∥   +i
    ↖ ∥ -1+i
    ← ∥ -1
    ↙ ∥ -1-i
    ↓ ∥   -i
    ↘ ∥ +1-i
    

**Turn Gates**

         ┌     ┐        ┌     ┐        ┌     ┐
    ↑⃞ = │ ↗ ↘ │    X⃞ = │ · → │   ↓⃞ = │ ↘ ↗ │
         │ ↘ ↗ │        │ → · │        │ ↗ ↘ │
         └     ┘        └     ┘        └     ┘

         ┌     ┐        ┌     ┐        ┌     ┐
    →⃞ = │ ↗ ↙ │    Y⃞ = │ · ↓ │   ←⃞ = │ ↘ ↘ │
         │ ↗ ↗ │        │ ↑ · │        │ ↖ ↘ │
         └     ┘        └     ┘        └     ┘

         ┌     ┐        ┌     ┐        ┌     ┐
    ↺⃞ = │ → · │    Z⃞ = │ → · │   ↻⃞ = │ → · │
         │ · ↑ │        │ · ← │        │ · ↓ │
         └     ┘        └     ┘        └     ┘

**Quordination Circuits**

Top Row

```
─X⃞────
─┴──↓⃞─

┌         ┐
│ ↗ · · ↘ │
│ · ↗ ↘ · │
│ ↘ · · ↗ │
│ · ↘ ↗ · │
└         ┘
```

Middle Row

```
─↑⃞──┬──→⃞─
────X⃞────

┌         ┐
│ → ↑ ↓ ← │
│ → ↑ ↑ → │
│ ↓ ← → ↑ │
│ ↑ → → ↑ │
└         ┘
```

Bottom Row

```
────┬──↑⃞─
─→⃞──Y⃞────

┌         ┐
│ → → ← → │
│ ↑ ↓ ↓ ↓ │
│ → ← → → │
│ ↑ ↑ ↑ ↓ │
└         ┘
```

Left Column

```
─↓⃞──X⃞────
────┴──↑⃞─

┌         ┐
│ → ↓ → ↑ │
│ ↓ → ↑ → │
│ ↑ → ↓ → │
│ → ↑ → ↓ │
└         ┘
```

Middle Column

```
─X⃞──┬──←⃞─
────X⃞────

┌         ┐
│ · ↘ ↘ · │
│ · ↖ ↘ · │
│ ↘ · · ↘ │
│ ↘ · · ↖ │
└         ┘
```

Right Column

```
────Y⃞────
─↑⃞──┴──→⃞─

┌         ┐
│ → ← ↑ ↑ │
│ → → ↓ ↑ │
│ → → ↑ ↓ │
│ ← → ↑ ↑ │
└         ┘

# ORB — 90-second pitch script

Two speakers, about 225 words. Stage cues in brackets.

## Chapter 1: Meet the team (0:00–0:10)

**Aarav:** Hey, I'm Aarav, CS at UCLA, and I worked on defense logistics software at Northrop Grumman.

**Parth:** I'm Parth, also UCLA CS. We started with one question: why does nobody actually know where their stuff is?

## Chapter 2: The problem (0:10–0:30)  [problem slide]

**Aarav:** Supplies move through a network, and every report about them comes from a person. A radio call, a text, a spreadsheet. Some are wrong, and nobody can tell which.

**Parth:** In Iraq, one point two billion dollars of supplies were shipped and never acknowledged. Ukraine aid was marked delivered weeks before it arrived. Water systems lose two trillion gallons a year to leaks nobody can find. Every tool today averages the reports, so one lie becomes a confident wrong answer. 

## Chapter 3: Live demo (0:30–1:10)  [switch to ORB]

**Aarav:** This is ORB. [click "view files" on Medical supply] Here's the raw input. Radio chatter, a convoy log, a spreadsheet. Different people, different formats, no schema.

**Parth:** [click the demo] One click. ORB reads every message, builds the network, and enforces one law: what leaves one site has to arrive at another.

**Aarav:** [graph appears] Seven sites recovered exactly. Two reports flagged. This one, a receiver claiming three-forty units, contradicts the driver's one-forty. Neither file could catch that alone.

**Parth:** [click view insights] And it doesn't stop at numbers. It briefs you: what happened, which site is degraded, who to call. [type in chat: "who do I call first?"] Ask it anything about next steps.

**Aarav:** And when the data can't decide, it says so before answering. It computes its own guarantee.

## Chapter 4: Why it wins (1:10–1:30)  [benchmark chart]

**Parth:** Least squares is what everyone ships today. On ten ground-truth scenarios it recovered the true state twice. ORB recovered it eight times, and the two it missed, it had already warned us about.

**Aarav:** Least squares blamed eight honest reports per run. ORB blamed almost none. It matches an exhaustive search of every possible lie, in one millisecond. Fuel, aid, water, same engine.

## Prep notes

- Run the Medical demo once before going on so extraction is cached and the click is instant.
- Click "view insights" while Parth is still on his line; the briefing takes a few seconds to stream.
- Have the chat question typed in advance so it is one keystroke.
- Slides: problem slide up through 0:30, ORB from 0:30, benchmark chart (docs/benchmark.png) at 1:10.

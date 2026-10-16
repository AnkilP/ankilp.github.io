---
title: Chesterton’s Wall
author: Ankil Patel
pubDatetime: 2026-10-16T20:00:00-04:00
slug: chestertons-wall
featured: false
draft: false
tags:
  - software engineering
  - AI
  - engineering practices
description: Most software engineering best practices are coping mechanisms for human cognitive limits. What happens to them when those limits move?
---

Most of what we call software engineering best practices are coping mechanisms for having a 1.4 kilogram brain. 

Like traditions that span religions, politics, professions and institutions, our software engineering rituals and practices are consequences of latent fundamental limitations. Small functions. Single responsibility. DRY. Code review. They're ritualistic accommodations for a processor that can juggle about four items in working memory (Miller's famous “seven, plus or minus two” turned out to be [generous](https://doi.org/10.1017/S0140525X01003922)), gets tired after a few hours, and can't reliably diff two files without tooling.

We’ve internalized these rituals so deeply, we’ve forgotten they’re contingent. The constraint shaped the practice but the practice outlived the constraint. Ask an engineer *why* functions should be short and you'll get post-hoc reasoning: testability, reusability, maintainability. All true. But missing the point. It’s really downstream of the real cause: humans can't hold large things in their heads - there are limits to our cognition and the amount of scenarios we can reason about.

We've been outsourcing our thinking to institutions since we could communicate. Take religious dietary rules about how meat is prepared: one common reading is that they encoded food-safety knowledge from before refrigeration. The original constraint dissolved, the practice persisted. Reasoning was lost. Traditions work because they encode valuable information about fundamental constraints that existed when the tradition formed. 

This isn't new. Software engineering has already weathered several constraint shifts. In fact, I’d argue that most engineering complexity can be removed if fundamental limits change meaningfully. 

We used to hand-optimize assembly because compilers were like, well, yesterday’s AI: stupid. Entire careers were built on knowing which instructions were faster on which architectures. Then compilers got better, and that expertise became much harder to find and train for. The practice of manual optimization didn't disappear, but it retreated to the edges: high-performance or extremely large scale systems, hot loops, embedded systems, the places where the new constraint (compiler intelligence) still had gaps.

We used to ration memory like it was precious, because it was. Data structures were chosen for space efficiency first. Then memory got cheap, and we started trading space for time without thinking about it. The engineers who'd learned to count bytes found their instincts actively misleading them.

![Log-scale chart of the inflation-adjusted price per terabyte of memory, flash, solid-state, and disk storage from 1956 to 2023, falling by roughly ten orders of magnitude](assets/images/memory_storage_prices.png)

*Source: [Our World in Data](https://ourworldindata.org/grapher/historical-cost-of-computer-memory-and-storage)*

Consider B-trees. Wide, shallow trees dominate disk-based databases because seeks are expensive. We minimize reads by packing keys per node and keeping the tree shallow. Red-black trees solve the same problem (ordered key-value lookup) but hold the tree in memory, allowing for deeper and narrower trees and more pointer chasing. Same abstraction, different structure, because the underlying constraint differs.

As memory got cheaper, entire database working sets moved into memory, and suddenly the B-tree ethos started to crack: the buffer pool, the page cache, the LRU eviction policies, the careful distinction between sequential and random I/O, the write-ahead log optimized for sequential appends, the checkpoint strategies… the whole apparatus for emulating memory with disk. A nontrivial chunk of it is compensation for the disk constraint. Databases like SingleStore started using skip lists from the beginning; why emulate random access with extra infrastructure when you have actual random access. I’d argue that much of the disk-based infrastructure was working around a hardware limitation and had little to do with any business problem. Remove the limitation and the infrastructure becomes dead weight. Engineers had to *decide*, consciously, how to adapt to new constraints rather than simply inheriting the disk-era default.

The B-tree story is about storage. But the same pattern plays out in compute.

We're seeing something similar in ML infrastructure. Once a model is too big for one accelerator's memory, you have to split it across chips, and how you split it depends on how fast those chips can talk to each other. Inside a GPU server, NVLink is fast enough for tensor parallelism. Across servers, the network is much slower, so past a certain scale engineers reach for pipeline parallelism, and the infrastructure burden explodes: microbatch scheduling, stage balancing, pipeline bubbles where stages sit idle waiting on their neighbours, and a lot of clever work overlapping compute with communication to hide them.

TPU pods are wired differently. Thousands of chips share a single high-bandwidth interconnect, which lets you stay in simpler parallelism modes much longer. Google trained the 540-billion-parameter [PaLM](https://arxiv.org/abs/2204.02311) across 6,144 TPU v4 chips without using pipeline parallelism at all. A good share of the pipeline-parallelism machinery exists to compensate for a constraint that doesn't bind as hard on different hardware.

In each case: a constraint relaxed, a practice became vestigial, and the engineers who understood *why* the practice existed adapted while others cargo-culted.

The constraint shifting now is cognition itself.

Not all of it. Not uniformly. But specifically: the limits on how much context a single "reviewer" can hold, how quickly code can be generated, how exhaustively execution paths can be traced. These are moving.

Here's what this looks like in practice. Last month a colleague asked me to stop splitting my PR because it was confusing the AI code assistant. I'd written scaffolding in one PR - functions, types, structure - with the integration coming in a follow-up. From the AI's perspective, I'd merged in code that nothing called. It kept getting confused about how my change related to a bug it was solving for another colleague. 

I'd staged the work into smaller pieces for good, well-established reasons:

1. human reviewers have limited attention. They miss bugs in large diffs. They get fatigued. Small PRs respect this constraint  
2. reverts need isolation. When something breaks at 3am, you want to undo one thing, not untangle a mega-commit. Small PRs make rollback surgical  
3. organizational scope control. Small changes are easier to reason about politically. They're easier to approve, easier to ship, easier to track

AI is loosening the first constraint. The second is changing slowly. The third hasn't moved. The practice of "small PRs" was never one practice. It was three constraints bundled together because they happened to point in the same direction. Now they don't.

This is what it looks like when bundles decompress. Consider code review:

* Catching bugs (cognitive → AI can help)  
* Knowledge sharing (organizational → unchanged)  
* Accountability and signoff (political/legal → unchanged)  
* Mentorship (developmental → unchanged)

If you think code review is "for catching bugs," you might conclude AI makes it obsolete. If you recognize it's a bundle, you realize only one leg of the stool is changing.

The same unbundling applies to documentation:

* Explaining what code does (cognitive → AI can generate this on demand)  
* Recording *why* decisions were made (organizational memory → still needed, maybe more)  
* Onboarding new team members (developmental → mixed)  
* Contractual/compliance requirements (legal → changing)

Some documentation becomes redundant. Some become more valuable precisely because AI can't infer intent from code.

And to modularity:

* Limiting what humans need to understand at once (cognitive → shifting)  
* Limiting blast radius of changes (operational → unchanged)  
* Enabling parallel work by different teams (organizational → unchanged)  
* Creating boundaries for ownership and accountability (political → unchanged)

The cognitive case for modularity weakens if AI can hold more context. The other cases don't budge.

So what happens next?

Adopting new practices fastest won't decide who navigates this well, and neither will throwing out the rulebook or clinging to it. The engineers who do well will be the ones who can look at an existing practice and ask: *what limitation does this exist to address? Is that limitation still real? For whom?*

Some practices will survive with new justification. Small functions might persist because they're easier for AI to regenerate and modify, long after humans stop needing them small. Same practice, different reason.

Some practices will become vestigial. Elaborate inline comments explaining what code does… AI can just tell you. The practice might linger through inertia, but the constraint is gone.

Some practices will invert. Consolidation might beat decomposition in contexts where AI needs semantic context more than humans need small diffs. Your instincts will mislead you.

Some practices will conflict with themselves. Small PRs for human reviewers versus large PRs with lots of context for AI assistants. You'll have to pick which constraint you're optimizing for, consciously, instead of assuming they're the same thing.

Best practices were always contingent, and we'd forgotten which traditions were predicated on which fundamental constraints. AI is just making that visible.

We've always outsourced cognition: to traditions, to institutions, to tools. AI is just the latest substrate. The more important question is whether we understand what we were outsourcing in the first place, and what happens when the thing we outsourced to gets smarter than the original constraint required.

Chesterton warned us not to tear down a fence until we understand why it was built. But he didn't tell us what to do when the cattle are gone and we're still maintaining the fence. Call it Chesterton's wall: the structure that persists after the constraint stops binding, because no one remembers why it was ever contingent.

The skill is knowing which walls are still holding the cattle back, and which ones are just blocking the view.

[image1]: (embedded image saved)
---
title: The Information Supply Problem
author: Ankil Patel
pubDatetime: 2026-10-16T18:00:00-04:00
slug: information-supply-demand
featured: false
draft: false
tags:
  - knowledge institutions
  - AI
  - misinformation
description: We ask questions faster than answers can be created. What fills the gap, and how are the new information institutions different from the ones they’re replacing?
---

Human curiosity generates an insatiable demand for information — one that traditional knowledge systems struggle to meet. Newsrooms, encyclopedias, research journals, textbooks — they all operate at a cadence that assumes questions arrive slowly, get vetted by editors, and get answered in units of articles or chapters. But we ask questions faster than answers can be created.

Where there’s a demand and no legitimate supply, someone will supply something anyway. That mismatch is fertile ground for misinformation — speculation and falsehoods filling in for the answers we don’t have.

Tools like ChatGPT help bridge this gap, but they do so in ways that are institutionally different from traditional news services and legacy knowledge providers. A newspaper has a masthead, a corrections page, and a legal department. An encyclopedia has editors and citations. A chatbot’s answer arrives with none of those attached — behind it sit training data, a system prompt, and a probability distribution. That’s a different kind of institution, with a different set of failure modes.

Whether these new tools are better or worse than the ones they’re replacing matters less to me than what happens to a society when the dominant information institutions no longer have the affordances that made the old ones legible: bylines, editorial responsibility, physical archives, retraction norms. Some of those affordances were slow and expensive. Some of them were also load-bearing.

## How the old institutions earned trust

It’s tempting to say we trusted newspapers because they were accurate. Mostly they were, but accuracy is hard for a reader to verify. What a reader could verify was accountability. A story had a byline, so you knew who to blame. It ran in a paper with an editor, so at least two people had looked at it. If it was wrong, the paper printed a correction in a known place, and if it was defamatory, the paper could be sued. Each of those is a mechanism that makes being wrong expensive for the publisher, and that cost is what a reader was actually relying on.

The second affordance is easier to miss: everyone read the same thing. When a paper got a story wrong, it got it wrong in public. Competitors could point it out, readers could write in, and the correction reached roughly the same audience as the error. A shared artifact made errors contestable.

The third is that claims were fixed in place. “Page 4 of Tuesday’s paper” refers to a specific set of words that will still say the same thing in twenty years. You can cite it, quote it, and argue with it, because it doesn’t change underneath you.

None of this was free. Editing takes time, corrections take time, and libel review takes lawyers. A good share of the slowness that left demand unmet was the price of these mechanisms.

## What a chatbot is, as an institution

Now look at an LLM through the same lens.

Its answers are generated on demand. There is no fixed artifact behind them; the text exists only once someone asks. And because models sample from a probability distribution, two people asking the same question can get different answers, sometimes materially different ones. There’s no “page 4” to point at.

Its errors are private. When a model gets something wrong, it gets it wrong for one person, in one conversation. Nobody else sees it, so nobody else can contest it. A newspaper error is a public event; a chatbot error is closer to bad advice from a friend.

Its corrections don’t propagate. If a lab fixes a failure in the next model version, future answers improve, but the people who already got the wrong answer never hear about it. Their notes, their emails, and the decisions they made on top of that answer stay as they were. There is no corrections page for a conversation that already happened.

And accountability is still being worked out in court. In 2024, a Canadian tribunal [held Air Canada liable](https://www.mccarthy.ca/en/insights/blogs/techlex/moffatt-v-air-canada-misrepresentation-ai-chatbot) after its website chatbot misstated its bereavement-fare policy; the airline had argued that the chatbot was a separate legal entity responsible for its own actions, which the tribunal called “a remarkable submission.” In the US, a radio host sued OpenAI after ChatGPT falsely said he had been sued for embezzlement, and a Georgia court [ruled in OpenAI’s favour](https://www.gibsondunn.com/gibson-dunn-wins-significant-victory-for-client-openai-defending-against-defamation-claim-based-on-hallucinated-generative-ai-output/). The rules are being written one case at a time, which is how you know they aren’t written yet.

## Which affordances were load-bearing

Some of what we’re losing was ceremony. Plenty of corrections pages were buried, and plenty of bylines were on stories nobody checked. The question is which pieces actually held the structure up.

My guess is the shared reference point. A society can argue productively about a fact only if people are looking at roughly the same fact. Social media already eroded this by giving everyone a different feed, but a feed is still made of public posts that someone else can see and rebut. A chatbot conversation is private by default. Misinformation used to be a broadcast problem, where one false claim reached many people and could be answered in the same channel. It’s becoming a one-to-one problem, where many slightly different claims each reach one person and no one is positioned to answer them.

Public correction matters for the same reason. A retraction is partly a fix and partly a ritual: it tells everyone the institution is willing to be wrong in public, which is what makes it worth trusting when it isn’t.

Attributable authorship matters less than I expected. Wikipedia has almost none in the traditional sense, and it became one of the most trusted references on the internet. What Wikipedia has instead is a public edit history, citations on every claim, and talk pages where disputes happen in the open. It answered questions far faster than a print encyclopedia ever could and kept most of the legibility anyway. That’s the most encouraging data point I know of: speed and legibility trade off, but less steeply than we assume.

## What new affordances could look like

If Wikipedia is the proof that the tradeoff can bend, the question is what the equivalent mechanisms look like for generated answers.

**Provenance by default.** An answer that cites its sources can be checked, and an answer that can be checked can be corrected. Retrieval-backed answers with inline citations already exist; the gap is making them the default rather than a mode you have to ask for.

**Fixed, shareable answers.** A permalink to a specific answer turns a private event into a public artifact. It can be quoted, disputed, and corrected, which gets back some of what “page 4” gave us.

**Changelogs as the new corrections page.** Labs already publish release notes for models. A version of that written for the public, saying which kinds of answers were wrong and are now fixed, would be the closest thing to a retraction a model can issue.

**Translation over vetted sources.** In [Tower of Babel](/posts/tower-of-babel) I argued for AI as a translation layer that lets information keep its shape while changing its presentation. That argument depends on what sits underneath. A model that translates from vetted, citable sources inherits their accountability. A model that generates from a probability distribution alone has none to inherit.

## The supply problem and the legitimacy problem

LLMs mostly solved the supply side. You can now ask almost any question and get a fluent answer in seconds, which is a remarkable thing and a real improvement over the speculation that used to fill the gap.

What they opened is a legitimacy problem. The old institutions were slow partly because they paid for the right to be trusted: editors, corrections, archives, liability. The new ones are fast partly because they haven’t paid yet. The same tradeoff keeps showing up across everything I write about — speed against legibility, flexibility against structure — and AI doesn’t make it disappear. It does give us a chance to rebuild the accountability mechanisms on purpose, instead of inheriting them by accident.

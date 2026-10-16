---
title: Provable Languages in Law
author: Ankil Patel
pubDatetime: 2026-10-16T12:00:00-04:00
slug: provable-language-in-law
featured: false
draft: false
tags:
  - law
  - language
  - AI
description: Can we reframe the law as a provable language?
---

India has roughly 21 judges per million people on paper, and closer to 14 per million actually seated. Its own Supreme Court, following a 1987 Law Commission recommendation, set the target at 50 per million and reaffirmed it in a 2002 judgment ordering the government to hit that number within five years. Nearly forty years after the recommendation and more than two decades after the court order, India is less than half of the way there — while the population it needs to serve keeps outpacing the sanctioned strength. For context, the US sits around 150 judges per million and Europe around 210.

Meanwhile, roughly [59 million cases](https://en.wikipedia.org/wiki/Pendency_of_court_cases_in_India) are pending across the Indian judiciary as of October 2026. More than 170,000 of them have been waiting for over thirty years. A 2018 Niti Aayog strategy paper estimated that clearing the backlog at the then-prevailing disposal rate would take 324 years, and that was when the backlog was 29 million cases. It has roughly doubled since. Justice, in the sense of "the state adjudicating a dispute within a useful timeframe," is not being delivered — and the ratio of pending cases to judges makes it clear that we cannot possibly hire our way out.

If we can't hire our way out, we have to redesign the system, and the place to start is the medium the law is written in. We wrote laws in natural language because that was the only substrate we had. That's no longer true.

Traffic law is where the compression is easiest to see, but it's also the least damaging case. If a red-light rule is too coarse — obey it unconditionally, even at 3am on an empty road — the cost is a few wasted minutes. Real damage lives in statutes that govern billions of dollars or a person's freedom: a tax code that has to fit in a book people can actually read, a criminal statute that has to leave room for a jury's judgment, a contract clause that gets litigated because the drafter and the reader parsed the same English words differently. The law's *intent* is usually clear in each of these. The trouble is that natural language forces the intent through a compression stage: the drafter has to imagine every case the statute will hit, pick words that cover the important ones, and hope the courts fill in the gaps. Every step of that pipeline leaks information.

Formal languages for law already exist, mostly in research labs. [Catala](https://arxiv.org/abs/2103.03198), developed at Inria, lets lawyers and programmers write a statute side by side with its executable version, paragraph by paragraph. Its authors used it to encode Section 121 of the US tax code and France's family benefits rules, and in the process found a bug in the French government's official implementation.

Here's what that looks like. Section 121 lets you exclude up to $250,000 of gain from the sale of your home from taxable income. For married couples filing jointly, the statute raises that cap:

> Paragraph (1) shall be applied by substituting “$500,000” for “$250,000” if—
>
> (i) either spouse meets the ownership requirements of subsection (a) with respect to such property;
>
> (ii) both spouses meet the use requirements of subsection (a) with respect to such property; and
>
> (iii) neither spouse is ineligible for the benefits of subsection (a) with respect to such property by reason of paragraph (3).

And here is the same paragraph in Catala, lightly abridged from the project's [US tax code examples](https://github.com/CatalaLang/catala-examples/blob/2694b17ddd7c2246e207e8f077d586349ee5abea/us_tax_code/section_121.catala_en#L269-L313):

```
scope Section121TwoPersons:
  rule section_121_b_2_A_condition
  under condition
    (return_type with pattern JointReturn)
    and
      # i)
    (
      section121Person1.requirements_ownership_met
      or section121Person2.requirements_ownership_met
    )
    and
      # ii)
    (
      section121Person1.requirements_usage_met
      and section121Person2.requirements_usage_met
    )
    # iii)
    and (not (section121Person1.section_121_b_3_applies))
    and (not (section121Person2.section_121_b_3_applies))
  consequence fulfilled

  exception
  definition gain_cap
  under condition section_121_b_2_A_condition
  consequence equals
    $500,000
```

Each numbered clause becomes one condition, and the comments point back to the clause it implements, so a lawyer can audit the code against the statute line by line. The `exception` keyword is the clever part. Statutes are written as a default followed by exceptions: the cap is $250,000, *unless* these conditions hold. Catala keeps that structure intact instead of flattening it into nested if-statements, which is where hand-written implementations of the law tend to drift from the text.

"Provable" is the important word. Once a statute is written formally, you can start checking properties of the law itself before anyone is subject to it: that no two provisions contradict each other, that every combination of facts produces an outcome, and that a given decision actually follows from the facts on record. In natural language, we find those gaps one lawsuit at a time.

AI fits on both sides of that formal core. On the way in, models can help translate existing statute books into formal code, which is the slow, expensive step that has kept tools like Catala in research labs. On the way out, they can turn messy evidence (bank statements, postal records, sensor logs) into the structured facts a formal rule needs. The rule itself stays explicit and checkable. That split matters. You don't want a model deciding what the law means, but you can let one do the clerical work of getting facts into a form the law can be applied to.

The same shift reaches well beyond courts. A self-driving car could follow a detailed, formal traffic code in place of rules simplified for human attention spans, and compliance could be checked as code is written instead of audited after the fact. And while we have the right to a trial, the backlog effectively starves out anyone not rich enough to hold out for one — the medium itself is part of the bottleneck.

Of course, this is not without risk, especially political risk. Who writes these machine-readable laws? Who decides what an algorithm considers fair? If AI personalizes the law itself by optimizing regulations in real time, does governance become hyper-responsive or hyper-political? If two people experience different tax policies, different contract terms, and different enforcement based on AI-optimized outcomes, do we get dynamic fairness or algorithmic feudalism? The ability to bridge the gap between intent and enforcement is a harbinger of power redistribution — one that could entrench hierarchies as easily as it could dismantle them.

Which brings me back to India's backlog. Not every case in it needs a judge's discretion. Around 4.3 million of them are cheque-bounce cases under Section 138 of the Negotiable Instruments Act, and in Delhi they make up roughly half of all trial court pendency. The statute is about as close to a formal specification as Indian law gets: a cheque is dishonoured, the payee sends notice within thirty days, the drawer fails to pay within fifteen. Most of those elements are facts you can check against a bank record or a postal receipt. Yet every one of these cases waits in the same queue as a murder trial, for the same scarce judges, because the law lives in prose that only a court can apply. Write that statute as a provable rule, let the evidence be checked automatically, and save human judges for the disputes that actually need judgment. That alone won't clear fifty-nine million cases, but no amount of hiring was going to either. Handled well, AI could do something we’ve never been able to: bend tradeoffs to our advantage instead of letting them dictate our choices.

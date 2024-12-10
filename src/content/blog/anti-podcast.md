---
title: High fidelity vs high understanding
author: Ankil Patel
pubDatetime: 2024-10-03
slug: anti-podcast
featured: true
draft: false
tags:
  - high fidelity
  - high understanding
  - tradeoffs
  - provable law
description: How can we preserve nuance without making knowledge inaccessible?
---

In an era of abundant information, we grapple daily with a difficult tradeoff: the tension between high fidelity (deep expertise, complexity, and nuance) and high understanding (clarity, approachability, and usability). We often compromise depth for breadth, or fidelity for accessibility. But what if we could find ways to balance both?

Patrick Collison tweeted about what makes great leaders. And his thoughts centered on three criteria:

- You need to have excellent judgment in your problem area
- You need to recognize the importance of good judgment as a phenomenon
- You need to demand it in others

He expands that leaders might fail on (2) or (3) because they might not be deep enough in their domains to be right on the substantive merits of questions within their purview (and unable to recursively detect/insist on that correctness in others, or to elevate and prize it when they see people who do it well).

Recently, I discovered I was suffering under the [Gell-Mann amnesia effect](https://en.wikipedia.org/wiki/Michael_Crichton#Gell-Mann_amnesia_effect) as I listened to a podcast that I had first hand knowledge in. The host was terribly underqualified: they just weren’t asking the right questions. And then I started seeing this in many places: news, conversations, magazines, radio shows… It seems like public discourse caters to the lowest common denominator because it's optimizing for accessibility not precision. Forget the ulterior motives like having a political agenda. Public discourse is lamentable because we have non-experts who don’t know what they don’t know ask the wrong questions or not ask the right questions. This is of course a blanket generalization as I've seen really phenomenonal journalism (hah how ironic, how could I, a non-journalist, know the difference). They are our bridge between high fidelity and high understanding. And this leads to people who don’t understand the deep institutional reasons for why certain phenomena exist the way they do. For example, most people believe that medical fees are high because hospitals are price gouging. But hospitals actually have very thin margins. Or it’s crazy that [dental and medical](https://www.theatlantic.com/health/archive/2017/03/why-dentistry-is-separated-from-medicine/518979/) practices are separated. But that’s because they have completely different organizations and schools of thought. There was an opportunity to combine them in the early days but ignorance kept it separated. Now, it’s going to be very hard to merge as both organizations want professional independence. These are the conversations I want to listen to. Not whether the next tech fad is going to raise unemployment. There was an interesting interview about net neutrality that gave the ISP's [perspective](https://www.youtube.com/watch?v=hKD-lBrZ_Gg) that seemed to explain that rationale behind the price hikes in a way that [John Oliver](https://www.youtube.com/watch?v=fpbOEoRrHyU) did not fairly present. These misconceptions thrive because nuanced discussions rarely penetrate mainstream discourse. Instead, we get accessible, simplistic stories that fail to address the deeper “why” behind the facts.

## A new discourse model

There is a gap in the public discourse. It's created because of this divide between fidelity and understanding. And it only exacerbates the asymmetry between the supply and demand of information. The goal is to be both truth-seeking **and** accessible. As I get all my life advice from cppcon: high fidelity impedes understanding. For far too long, we as a species have had to contend with this tradeoff between high fidelity and high understanding. I want a good search tool (maybe AI) to supplement the lack of understanding so that we can extract value from high fidelity conversations without needing to be in the trenches or deeply understand the material.

I envision a platform fostering debates between domain experts, stripped of speeches and soundbites. Imagine Marc Andreessen debating Lina Khan, or Paul Krugman engaging Niall Ferguson. Both might reach opposing conclusions, but their [intellectual rigour](/posts/conspiracy) way could elevate public understanding. The closest format imo is the [Munk Debates](https://munkdebates.com/), which are only held in Toronto twice a year and the hosts have to address the audience's sensibilities. What if we innovated further?

- Written debates: Allow participants time for well-researched arguments
- Contextual panels: Experts who clarify abstractions, provide historical context, and address audience questions
- AI-assisted tools: To bridge gaps in comprehension and synthesize high-fidelity content into actionable insights

This approach could shift discourse away from the lowest common denominator, fostering both truth-seeking and accessibility.

## A provable, formal language for law

The high fidelity vs. high understanding tradeoff also plagues legal systems: we have lawyers and judges (depending on the legal system) to help us interpret the law, which is really a utilitarian proxy for ethical behaviour. For example, waiting for the green signal is good because it creates predictable traffic, which means fewer people get hurt and more traffic moves through the city. But traffic lights come at a personal cost to each individual that has to wait. So if its late at night, would you cross a red signal if no one's around? Most rules are proxies for how we should behave but in many cases, we can optimize further without hurting anyone. However, laws that are too specific are confusing and people tend to miss the larger picture. Our laws are uniquely human - if it's too specific, it's hard to follow and if it's too vague, it's hard to enforce. The preciseness of the law has to be tuned to our ability to follow them. What if there was a formal, provable language that lets us specify scenarios and several high level objectives (utilitarian + egalitarian, etc) and have this system provably give a judgment on crimes/violations and relieve our court systems (blog post coming soon with more details, including existing approaches like [Catala-lang](https://catala-lang.org/))? That could let us build agents, like self-driving cars, that could follow high fidelity laws and let us express exactly how we want the world to look in a way that's more accurately and precisely.

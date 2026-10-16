---
title: Tower of Babel
author: Ankil Patel
pubDatetime: 2025-01-13T00:14:09-06:00
slug: tower-of-babel
featured: true
draft: false
tags:
  - bureaucracy
  - standardization
  - productivity
  - embedding models
  - translation
description: Tradeoffs have shaped human civilization. Every law, every institution, every economic rule is a compromise between structure and flexibility, order and adaptation. But can we free ourselves from some of the most fundamental tradeoffs?
---

Somewhere above 35,000 feet, I found myself in an unexpected debate with a partner at a major accounting firm. His firm, like many others, had embraced AI-powered tools to digitize receipts with just a camera scan.

I wondered how much time, energy and effort went into this process. Data has to be collected, cleaned, and structured. AI models must be trained, deployed, and continuously monitored for relevance. Now contrast that with parsing a well-structured receipt. Every invocation of an AI model requires appreciably more energy than parsing a structured response. Why do companies invest in high capex, energy-intensive AI workflows to solve a problem that could be eliminated at the source? How would AI ever compete with standardization?

It raised a question: Why don’t we just standardize receipts so we can skip the digital gymnastics altogether?

He chuckled. “We’re not in the business of pushing new standards.”

That remark stuck with me. Why do we suffer inefficiency when the alternative is so seductively elegant? 

The large variance in receipt formats persists because of inertia, misaligned incentives, and the natural complexity of large-scale systems. In theory, businesses should welcome streamlined formats that reduce costs and errors. In practice, switching to a new standard or creating a new standard is costly, and no single entity wants to bear that burden alone. Even when the benefits are obvious, standardization is hard to justify unless there’s a clear economic or strategic advantage.

Some standards, like the USB port or the metric system, unlock innovation by creating common ground. Others, like proprietary file formats or walled-off ecosystems, reinforce dominance by making alternatives costly. The question of standardization is about who gets to decide, what tradeoffs are baked in, and how much room remains for evolution.

Historically, standardization has played a crucial role in trade and the exchange of ideas. A shared language lowers barriers, improves coordination, and accelerates commerce. But standardization has always come with tradeoffs. Take the French suppression of Occitan: for centuries, Occitan thrived as a literary and cultural language across southern France. But as the French state centralized power, it imposed Parisian French as the national standard, systematically discouraging and even banning Occitan in schools. This made administration and governance more efficient, but it also erased a linguistic identity that had flourished for generations.

For generations, we solved coordination problems with standardization: roads, laws, languages, regulations, protocols. Every standard is a blunt instrument, enforcing order at the expense of adaptability — a Faustian bargain that reduces chaos at the cost of flexibility. Between structure and fluidity, precision and accessibility, efficiency and resilience, every system we build is a compromise. But what if we could have both, without forcing artificial constraints?

Tradeoffs are fundamental to how complex structures emerge. Cities, markets, ecosystems… none are truly centrally planned, yet they function through local interactions that self-organize into global patterns. When constraints are adaptive instead of rigid, systems become more resilient, more intelligent, more alive.

Technology has consistently created slack to alleviate the effect of tradeoffs where competing priorities, technical limitations, and institutional inertia grind progress to a halt. Often the solutions already exist and the bottleneck is that different stakeholders optimize for different things. Businesses want interoperability, but they also want proprietary advantages. Consumers want convenience, but they also (sometimes) want privacy. Every system is shaped by the tug-of-war between those who shape its incentives, as much as by technical constraints. Just as previous technological shifts redefined what was once considered an unavoidable compromise, AI will erode some of our most fundamental tradeoffs, replacing rigid binaries with adaptive, context-aware solutions.

The mechanism I have in mind is a learned shared representation. Today, getting N systems with N formats to talk to each other takes either a standard everyone agrees to, or up to N×(N−1) pairwise converters, one for each direction. A model trained on many formats offers a third option: a translation layer. It maps each format into a common internal representation (an embedding) and can map out of that representation into any format it knows, so each system needs one way in and one way out. Nobody has to negotiate the shared format, because the model learns it from data.

There's precedent for this in machine translation. In 2016, Google trained a [single model on many language pairs](https://arxiv.org/abs/1611.04558) and found it could translate between pairs it had never seen together: trained on Japanese⇄English and Korean⇄English, it could translate Korean⇄Japanese directly. The model had learned something like an interlingua. Receipts, invoices, and insurance claims are a harder case than prose, since a wrong digit matters more than an awkward phrase, but the idea carries over. The standard still exists. It lives inside the model instead of in a spec everyone has to adopt, and the output side can adapt to each recipient. In principle, that lets businesses, institutions, and entire economies avoid choosing between fragmentation and rigid standardization.

Today, though, we mostly point AI at rigid systems instead of using it to rethink them. The more interesting question is whether AI can loosen the grip of compromises we once saw as intrinsic to how the world works.

Take the standardization tradeoff between chaos and flexibility. Can we use AI as a translator to get the benefits of standardization while mitigating its risks? What if, instead of enforcing a single format, we allowed each system to speak its own language? Consider spoken language: we don’t force the world to speak a single tongue for efficiency’s sake. Instead, we translate.

![A photo of a cat captioned "The cat is red." in English, translated to "Le chat est rouge." in French while the photo stays the same](assets/images/cat.png "Some insist the cat is orange")

Consider how we consume information today. We all receive the same receipts, the same websites, the same ads - fixed formats designed for the majority. But if you and I consume information differently, why do we see the same websites? Some prefer reading, others listening, others skimming structured data or if you’re like me, you prefer allegorical folktales to explain how the world works. Can we decouple information from its presentation?

Imagine getting a receipt that isn’t locked into a fixed PDF. The underlying data travels with it, and a model renders it however you need: audio if I prefer to listen, a table of raw line items if you’re reconciling expenses. The information is the same, but its presentation shifts to fit the user. It’s a fundamental shift in how we interact with the digital world. And once you accept that shift for a single receipt, the same logic scales all the way up to entire industries.

Right now, companies spend vast amounts of time and money just making their systems talk to each other. Supply chain transactions, insurance claims, financial reports - every industry has its own formats, its own schemas, its own labyrinth of integrations. A retailer receives invoices in one format, but their accounting software expects another. A hospital shares medical data, but insurers demand it in a completely different structure. The result? A never-ending cycle of middleware, APIs, and translation layers, each solving a problem that shouldn’t exist in the first place.

![Two network diagrams: on the left, banks, insurers, hospitals, and agencies wired to each other point-to-point; on the right, each connects only to a central AI translation layer](assets/images/translator.png "")

Left: Fragmentation meant more middleware, more APIs, more headaches 

Right: AI offers a different path — a bridge rather than another layer

AI here becomes connective tissue: shifting the burden of translation from people to machines could make supply chains more resilient, financial regulations more adaptable, and healthcare records more accessible. Information stops being rigid and starts being fluid, adaptable, and deeply personal.

Zoom out further and this stops being a story about receipts, or even about interoperability between companies. Fragmentation is what happens whenever many independent actors coordinate at scale, and that describes an entire economy. So the question becomes: what does a translation layer look like at that altitude, and where would it matter most?

The U.S. economy is a good place to start, because it thrives on a juxtaposition: the same free-market policies that fuel competition and resilience also leave healthcare, finance, government, tech, insurance, and education operating in fragmented silos. Locally sub/optimal, holistically nonoptimal. Call them Canadian sensibilities but I was bewildered by American institutions and their success despite the chaos — until I realized the chaos was the price of the resilience.

Consider the healthcare industry. It’s famously fragmented, with private insurers, state-specific regulations, and myriad providers all doing things their own way. It’s incredibly frustrating for patients who have to navigate a maze of forms and coverage rules. But it’s also the same ecosystem where new surgical technologies, experimental treatments, and novel financing models can pop up rapidly (vs Canada). The messy, unregulated edges sometimes foster fresh ideas that might be stifled in a more centralized system.

So is there a way to retain the dynamism of the free market without inheriting its inefficiency? Some countries already structure their economies to minimize fragmentation through governance rather than technology. What can we learn from them, and is that something Americans et al. should aspire to?

If we look at smaller or highly centralized nations, e.g. Estonia or Singapore, efficiency is baked into their institutions. They have strong e-government infrastructures, streamlined public services, and top-down coordination that keeps fragmentation in check. These countries might not even feel the need for an AI “bridging function” quite as urgently, because they already minimize fragmentation through policy and planning. But in a place like the US, where the fragmentation is deeply embedded, an AI translator could turn that free-market-borne fragmentation into an advantage by allowing coherence without losing agility. 

While AI can help fragmented economies like the U.S. regain coherence, developing nations face a different choice: they can avoid fragmentation internally by designing digital-first infrastructure from the start.

Consider how Kenya’s M-Pesa revolutionized financial access by skipping the existing banks entirely and moving straight to mobile payments. Or how some countries have built national digital ID systems that integrate seamlessly with government services, while antiquated bureaucracies like Germany’s still wrestle with snail mail, paper-based records and fragmented databases. The leapfrog potential is enormous: by designing for interoperability from the start, they can avoid the legacy inefficiencies that wealthier countries now struggle to undo.

However, these countries still have to trade and interact with countries with varying levels of fragmentation, making the bridging function an inevitable necessity.

So the translation-layer idea plays out differently depending on where a country sits — but nowhere does it make the underlying tradeoff go away. Fragmentation is a fundamental reality of human systems, well beyond just a technological or economic challenge. Every attempt to unify, from trade agreements to digital protocols, comes with its own tensions, baked into the architecture of our institutions.

The recursive irony is that explaining this compromise requires its own tradeoff. Even what you’re reading right now is a tradeoff between fidelity and accessibility! To make these ideas accessible, I’ve sacrificed nuance. For example, an OCR expert would remark on how VLM (vision-language models) struggle with reading receipts in ways that modern OCR would not. Or an economist like Ronald Coase might argue that AI would simply shift tradeoffs, not eliminate them. Or an AI researcher might similarly argue that if two systems interpret the same AI-generated representation slightly differently, do we simply shift the burden of misunderstanding rather than removing it? Or an engineer might ask if using AI as a translator just creates more standards and two parties, nevertheless, have to negotiate which AI model version to use? But that’s exactly the point: some tradeoffs are an unavoidable reality of human decision-making. 

![xkcd comic 927: with 14 competing standards, someone proposes one universal standard, and now there are 15 competing standards](assets/images/xkcd_standards.png "A standard (haha) retort to standard-making, courtesy of xkcd")

(Courtesy of xkcd)

None of this makes the translation-layer bet less interesting; it just means the bet moves the tradeoff somewhere else.

And yet, for all its transformative potential, my humble opinion is that the broader AI discourse remains uninspired. Regulators fixate on containing big tech on one end while simultaneously trying to maintain an edge over geopolitical rivals, effectively stranding themselves on a tightrope. However, big tech, in many aspects, also lacks imagination as they treat AI as a mere productivity tool when it could be an architectural one — a tool for reshaping the world as it could be, rather than only optimizing the world as it is.

Every generation has inherited its predecessors' tradeoffs. AI won’t erase ours, but for the first time we have a tool to reshape them. If we use it merely to optimize old institutions, we will have built a machine to calcify the past.

We can probably build a more fluid, emergent system. The real question is whether we have the vision and will to do so.

*Two threads I pulled out of an earlier draft of this essay and gave their own posts: [rewriting the law itself as a provable language](/posts/provable-language-in-law), and [the mismatch between our demand for information and the institutions that try to supply it](/posts/information-supply-demand).*

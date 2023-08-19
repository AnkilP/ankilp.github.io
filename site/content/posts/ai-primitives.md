---
title: "AI Primitives in Distributed Systems"
date: 2022-10-11
draft: false
---

## Motivation

I asked my former manager, Scott Wisniewski, about how to build resilient systems and his response was that I had the wrong idea. The holy grail wasn't in resilient systems - it was anti fragile systems (from the book by Taleb). Anti-fragile systems are able to learn and become better with new stressors. How do we build antifragile systems? AI provides an avenue for creating agents that can adapt to new information and learn more efficiently than previous techniques. 

When I search for AI in distributed systems or AI in large scale systems, I see [articles](https://engineering.fb.com/2021/07/15/open-source/fsdp/), papers and blogs on how to create systems that can train or do inference - i.e. create systems that can help create models. But I've only seen a few instances on how to use AI to boost the performance of isolated distributed systems. Interestingly, they were written by *interns* at [Jane Street](https://blog.janestreet.com/what-the-interns-have-wrought-2020/) and Databricks or academics who later moved to Jane Street and Databricks (coincidence, I think not) suggesting to me that this idea has enough merit to investigate and potentially enough that companies are willing to put their weight behind it. I'm not talking about augmentation of distributed systems, e.g. forecasting of traffic to provide signals to an autoscaler but rather I'm talking about something more in the vein of [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35). Software 2.0 focuses on the individual services while I'm proposing a different approach for systems by focusing on system aspects like antifragility.

I understand why it's been designed this way: classical control theory has rigorously proven theorems and established methods of achieving stability and correctness - hence why kubernetes is still using controllers invented in 1900s. Interestingly, when I asked in my reinforcement learning course what the difference was between reinforcement learning and control theory, the professor's response was that the two research groups just don't meet despite solving very similar problems. My aim is to replace some of these 'control theory' controllers with 'reinforcement learning' controllers, partly because we've seen how compact and simple RL controllers can be while emulating the same behaviour: [https://www.deepmind.com/blog/accelerating-fusion-science-through-learned-plasma-control](https://www.deepmind.com/blog/accelerating-fusion-science-through-learned-plasma-control)

For a business serving customers, stability and correctness translate directly to customer retention. However, these needs have changed and even startups can start to ingest thousands of GBs of data every day, serve thousands of customers within a relatively short period of time, all while battling cloud costs in the name of growth. If our needs have changed, why haven't the algorithms underpinning the modern internet changed?

A part of me wondered how many of the system-level issues we saw came from poor designs, changing environments that were no longer suitable to the implementation, terrible metrics (or lack thereof) and intuitive decision-making vs data-driven processes. Unfortunately, most of these problems are endemic of the entire industry, partly because of inertia, lack of infrastructure and de-prioritization because customers don't care. Why must we limit ourselves to such rigid system primitives? Instead, what if we had differentiable, tunable and *intrinsic* system primitives that were exposed to a learning agent. To me, chaos testing is to systems as what sanitizers are to C++ in that both are afterthoughts. Why can't we build systems that intrinsically exploit stressors to adapt to them in the same way that we should just use rust? 

I'm going to explore what AI primitives are by introducing a few examples. 

### Load Balancing

The goal of load balancing is to make sure that we distribute requests, jobs and workloads to all available server, workers and buckets in a way that doesn't overwhelm any one server. We can do this with consistent hashing or rendezvous hashing. These aren't cryptographic hash functions so we can use a hash function that's parallelizable and pipeline-able, so long as we can bound entropy on each bucket. However, what if I ask you to implement anti-affinity, affinity, sparsity and capacity (e.g. heterogeneous compute)? Current methods can't embed all that information into the hash function, even with virtual nodes. It requires conscious thinking on the engineer's part to implement those features.

However, there are ideas from [image retrieval](https://arxiv.org/pdf/2101.11282.pdf) or general [kernel-based hashing](https://ieeexplore.ieee.org/document/6247912) on creating kernels that can embed that information. The basic idea is that you have a vector space of load balancing configurations where each vector corresponds to a hashing function. Instead of minimizing entropy during training, we can maximize entropy - e.g. images/requests that are more unlike each other are placed in the same bucket. We can create hash functions from these kernels and use them with consistent or rendezvous hashing.

If this system is isolated, it doesn't warrant any of this work. With a holistic view of the system (section coming later), you can redirect requests based on how downstream services are tuning their traffic shaping, how backed up downstream queues are or which versions are being deployed. This can expand to the entire network topology.

This differentiable kernel can be interpreted as an embedding in a configuration space. Why is it important that it is differentiable? So we can take feedback from servers and tune the kernel as it receives feedback in an efficient way. Non-differentiable functions are harder to optimize. How is this different from a config file that we treat as a state/embedding? A config file has very specific assumptions about the system, the types and the format. More importantly, it's not an intrinsic part of the system, it's just an interface to the system. 

Oh, but we can do more with these embeddings. Suppose we have a hundred load balancers, each with its own set of parameters, traffic distributions and environments. And now, I come to you and say I want another load balancer but I don't want to train this hash function again. Well, what if we transport all these embeddings into a [super embedding space](https://neurips.cc/virtual/2021/workshop/21833) that's agnostic to implementation details. Then, we can query this super configuration space for another load balancer kernel given some parameters and then run it through a decoder network to get something implementation specific. 

### Query Optimizers

When queries aren't optimized correctly, it can introduce delays across all functions. Solutions like [airops](https://www.airops.com/blog/using-ai-to-optimize-your-sql-queries) and [postgres.ai](https://postgres.ai/products/joe) already exist but they're not first class primitives. They're interfaces to the query optimizer, which [need to be tediously maintained, especially as the system’s execution and storage engines evolve](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf). I'd argue that there are benefits to understanding the history of calls to a database to exploit caches and improve index selection. [Neo](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf) proposes a neural network based query optimizer that is comparable to optimizers from Microsoft and IBM. More importantly, it can dynamically adapt to [incoming queries, building upon its successes and learning from its failure](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf). Does this neural network fit the model of an system primitive? By shifting the weights of this neural network, we can bias different heuristics and come up with widely different outcomes based on the environment, storage needs, and traffic distribution. 

System primitives can come in various shapes and formats - the invariance is that they can be learned and adjusted by an AI agent (section coming later). 

### Compilers

I've often argued that if we can trust code written by an AI co-pilot, we might also be able to trust an IR compiled by an AI. Being able to compile code against heterogeneous compute proves to be a difficult task, partly because code is not always written with the hardware or scheduler in mind. For example, compilers don't natively account for memory models in RPC calls, sending compute to GPUs and cache affinity. I think of compilers as transformer models with restricted input/output. 
https://www.phoronix.com/news/Cluster-Scheduling-Hybrid-6.6
https://www.phoronix.com/news/System76-Scheduler-2.0.1

### Sharding

Sharding relocates certain rows and columns in a database to make it easier to scale read and write workloads. However, with more complex data queries which include joins, database caches, external caches sitting on top of databases, and frequency of changing values and quantities, sharding becomes a tough optimization problem. 

Like load balancing, sharding is also an allocation problem. However, we are trying to group rows or columns together in a k-means-like approach, where k can be different in various places - i.e. the size of the shard can be non-uniform. You can use a similar kernel as before but trained on a different set of data. Re-sharding can be very expensive so we would want to bound the deviation in the kernel or have a separate agent (section coming later) that could include the cost of re-sharding in its cost function. 

The key on which we shard also depends on how we're using these databases. That brings me to knowledge graphs (section later) - can we query system history from a knowledge graph to make informed decisions about how to bias the sharding kernel?

## Knowledge Graphs

What if we could index everything, or more practically, sample everything, that the organization has seen: requests, anomalies, issues, pages, etc. and build a knowledge graph out of it? Companies don't always understand the full capabilities of monitoring or dashboards - it's just 'report-the-news' whereas a more 
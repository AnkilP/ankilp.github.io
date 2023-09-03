---
title: "Anti-fragility in (Distributed) Systems"
date: 2023-08-30
draft: false
---

## Motivation

I asked a former manager (I haven't asked him yet if I can use his name in this blog post) about how to build resilient systems and his response was that I had the wrong idea. The holy grail wasn't in resilient systems - it was anti fragile systems (from the book by Taleb). Anti-fragile systems are able to learn and become better with new stressors. How do we build antifragile systems? AI provides an avenue for creating agents that can adapt to new information and learn more efficiently than previous techniques. 

A part of me wondered how many of the system-level issues we saw came from insufficient designs, changing environments that were no longer suitable to the implementation, terrible metrics (or lack thereof) and intuitive decision-making vs data-driven processes. Unfortunately, most of these problems are endemic of the entire industry, partly because of inertia, lack of infrastructure and de-prioritization because these issues aren't visible to customers. 

When I search for AI in distributed systems or AI in large scale systems, I see [articles](https://engineering.fb.com/2021/07/15/open-source/fsdp/), papers and blogs on how to create systems that can train or do inference - i.e. create systems that can help create models. But I haven't seen many instances on how to use AI to boost the performance of end-to-end systems in production. [Jane Street](https://blog.janestreet.com/what-the-interns-have-wrought-2020/), Databricks and [Google](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview) have invested in using AI in their infrastructure but I largely see this idea discussed in academia suggesting to me that this idea has enough merit to investigate and potentially enough that companies are willing to put their weight behind it. Note, I'm not talking about augmentation of distributed systems, e.g. forecasting of traffic to provide signals to an autoscaler but rather I'm talking about something more in the vein of [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35). I'm not sold on an empirical and declarative way of writing software, however, I could be convinced of the benefits by framing optimization problems as declarative problems, like with [Predibase](https://predibase.com/).

Software 2.0 focuses on the individual services while I'm proposing a different approach for building systems by focusing on system aspects like antifragility. My first thought was to create a learning agent to adjust parameters on a config file. A config file has very specific assumptions about the system, the types and the format. However, it's not an intrinsic part of the system, it's just an interface to the system. It's easier to sell and integrate into tech stacks, but we need to make assumptions about the behaviour of the underlying service and integrity of the interface (e.g. fields in the config that get ignored in the code). This suggests to me that config files are really second class system primitives. The distinction between a first class and second class primitive is that a first class primitive is an intrinsic aspect of the system and *can evolve with the system*. An interface to the system is more of a proxy and it requires collaboration between teams and considerably more re-training. Moreover, it can involve shims and careful engineering work to migrate to new versions.

I understand why it's been designed this way: classical control theory has rigorously proven theorems and established methods of achieving stability and correctness - hence why kubernetes is still able to use controllers invented in the 1900s to scale to large workloads. When I asked my reinforcement learning (RL) professor what the difference was between reinforcement learning and control theory, his response was that the two research groups just don't meet despite solving very similar problems. Even Sutton and Barto in their seminal RL book talk about how control theory is a [precursor](https://ai.stackexchange.com/questions/11375/what-is-the-difference-between-reinforcement-learning-and-optimal-control) to RL but have diverged because of their differing objectives. It seems like [data-driven control theory](https://fab.cba.mit.edu/classes/865.21/topics/control/07_reinforcement_learning.html) is going to use more modern ML techniques and the line between these two fields gets more blurry. My preference for RL is not motivated by capability, because it seems that gap is getting smaller, but rather by simplicity of the model because we've already seen how compact and simple [RL controllers](https://www.deepmind.com/blog/accelerating-fusion-science-through-learned-plasma-control) can be while emulating the same behavior as control-theory controllers with more generality. 

For a business serving customers, stability and correctness translate directly to customer retention. However, these needs have changed and even startups can start to ingest thousands of GBs of data every day, serve thousands of customers within a relatively short period of time, all while battling cloud costs in the name of growth. On top of that, when systems inevitably break because some of config issues, unexpected traffic distributions, latent system bottlenecks, etc. we tend to assign multiple engineers to fix those issues, costing the business money and, what is infinitely more valuable, time. If our needs and capabilities have changed, why haven't the systems underpinning the modern internet changed?

Why must we limit ourselves to such rigid system primitives? Instead, what if we had differentiable, tunable and *intrinsic* system primitives that were exposed to a learning agent? I want to sum up my sentiment with this analogy: chaos testing is to systems as what sanitizers are to C++ in that both are afterthoughts. Why can't we build systems that intrinsically exploit stressors to adapt to them?

## Hierarchal Autonomous Agents

I've hinted throughout this blog post that we could realize global efficiencies across the entire system by having individual AI primitives coordinate with a hierarchal AI agent that could optimize objectives spanning across the entire system. We have effectively created a [multi-agent](https://arxiv.org/pdf/1911.10635.pdf) system where the agents must cooperate to realize their common goals. 

For brevity, I will call this top-level AI agent: central controller (CC). 

I'd like to motivate the need for the CC with an example exemplifying the system level changes that are incurred when a new endpoint is added. The new endpoint is serving a small group of customers initially who are read-heavy. Maybe we should re-shard so that we can handle the new number of read requests? Or maybe instead we should create a cache in front of the database and schedule the polling of the database to validate the cache based on historical data of when the database isn't used as much? Or maybe the current system can handle the traffic but the system can exploit database caches if it ensures that the read requests from the new endpoint are issued from the same nodes that the write requests are assigned to? How do we decide which of these solutions is globally better? What objectives are driving decisions at organizations: money saved? strategic positioning? reducing complexity? Eventually, it'd be great to be able to train joint text-policy embeddings so we can ask the CC, in natural language, to optimize on a set of objectives - yet another use case that accessible LLMs have made possible. 

It's open to interpretation whether we want the CC to directly design other agents' policies.  

## System/AI Primitives

Generally, a lot of system-level problems can be framed as some optimization problem and I'm going to explore what AI primitives are by introducing a few examples that solve assignment, scheduling and search problems. Note that these are just a subset of what we can build.

## Assignment Problems

### Load Balancing

The goal of load balancing is to make sure that we distribute requests, jobs and workloads to all available server, workers and buckets in a way that doesn't overwhelm any one bucket. We can do this with consistent hashing or rendezvous hashing if we expect any of the nodes to experience downtime but the **basic idea is to facilitate some sort of an assignment problem**. These aren't cryptographic hash functions so we can use a hash function that's parallelizable and pipeline-able, so long as we can bound entropy on each bucket and on the collective. However, what if I ask you to implement anti-affinity, affinity, sparsity and capacity (e.g. heterogeneous compute)? We want to embed all that information and other biases into the hash function to direct requests to certain nodes. But what if we can use empirical data to tune the hash function based on short- and long-term trends. Even large companies like Ali Baba (I can't find the article but they describe running into hot spotting issues during Christmas despite designing a [resilient system](https://note.com/kenwagatsuma/n/n33f1caaab7cd)) have had issues during Black Swan events.

There are ideas from [image retrieval](https://arxiv.org/pdf/2101.11282.pdf) or general [kernel-based hashing](https://ieeexplore.ieee.org/document/6247912) on creating kernels that can embed that information. The basic idea is that you have a vector space of load balancing configurations where each vector corresponds to a hashing function. For this specific load balancing problem, instead of minimizing entropy during training, we can maximize entropy - e.g. images/requests that are more unlike each other are placed in the same bucket. We can create hash functions from these kernels and use them with consistent or rendezvous hashing.

The real benefits of the system only become clearer with a holistic view of the system - you can redirect requests based on how downstream services are tuning their traffic shaping, how backed up downstream queues are or which versions are being deployed. This thinking can expand to the entire network topology. What if we could have an AI agent tuning this kernel, and consequently redirecting requests or jobs, based on dynamic shifts in traffic and system health across the organization? 

This differentiable kernel can be interpreted as a system primitive (embedding) in a configuration space. Why is it important that it is differentiable? So we can take feedback from various signals and tune the kernel in an efficient way and because non-differentiable functions are harder to optimize. Kernels are just one interpretation of AI primitives. We can think of this assignment problem more abstractly: it could also be a RL agent that learns how to send requests to the right node depending on system characteristics. And the policy that it learns from interacting with the system can be a system primitive. We can create hierarchal RL agents that talk to each other and optimize different parts or dimensions of the system as a whole.

Suppose we have a hundred load balancers, each with its own set of parameters, traffic distributions and environments. And now, I come to you and say I want another load balancer but I don't want to train this hash function/RL agent/primitive again. Well, what if we transport all these embeddings into a [super embedding space](https://neurips.cc/virtual/2021/workshop/21833) that's agnostic to implementation details. Then, we can query this super configuration space for another load balancer primitive given some parameters and then run it through a decoder network to get something implementation specific. The great part about this super embedding space is that it can be reused across organizations, companies and scales.

### Sharding

Sharding relocates certain rows and columns in a database to make it easier to scale read and write workloads. However, with more complex data queries which include joins, database caches, external caches sitting on top of databases, and frequency of changing values and quantities, sharding becomes a tough optimization problem. For example, if there's a thundering herd targeting a specific group of shards, we might want to be able to reorganize shards to be capable of addressing those scaling issues while we're serving users and without manual intervention. 

Like load balancing, sharding is also an assignment problem. However, it differs from load balancing in how the underlying model is trained. Re-sharding can be very expensive so we would want to bound the deviation in the kernel and include the cost of re-sharding in its cost function. Moreover, the frequency at which we change parameters would be drastically different from how we might change the load balancer. 

The key on which we shard also depends on how we're using these databases. A localized view of stressors might create a tuning that doesn't address long-term trends. And it doesn't account for human intuition about human-centric traditions like Christmas-time load. An empirical model trained with 11 months of data with slight disruptions might never understand why the system becomes very busy on the 12th month. Or it might consider the 12th month as a anomaly.

## Search Problems

### Query Optimizers

When queries aren't optimized correctly, it can introduce delays across the system because databases underpin a lot of systems. Solutions like [airops](https://www.airops.com/blog/using-ai-to-optimize-your-sql-queries) and [postgres.ai](https://postgres.ai/products/joe) already exist but they're not first class primitives. They're interfaces to the query optimizer, which [need to be tediously maintained, especially as the system’s execution and storage engines evolve](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf). There are benefits to understanding the history of calls to a database to exploit caches and improve index selection. What does a first class query optimizer primitive look like?

[Neo](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf) proposes a neural network based query optimizer that is comparable to optimizers from Microsoft and IBM. More importantly, it can dynamically adapt to [incoming queries, building upon its successes and learning from its failure](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf). Does this neural network fit the model of an system primitive? By adjusting the weights of this neural network, we can address novel situations based on the environment, storage needs, and traffic distribution. 

System primitives can come in various shapes and formats - the invariance is that they can learn. I will add that they can also communicate and interact with an AI agent. 

## Scheduling Problems

### Schedulers

So far, we've talked about creating first class system primitives for assignment and search problems. What about scheduling problems? What about scheduling and assignment problems? When you submit jobs to your cluster, you request system resources like compute and memory. But the system doesn't precisely know how much compute and memory you'll need in the duration of your program execution. What happens if there's a large burst of ML training jobs and you need to quickly provision compute and memory for those jobs? You could over-provision and waste resources or under-provision and redo or slow down the task. There are a few [examples](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview) that have already used AI to find a better packing solution. 

We also see scheduling problems in process schedulers for [hyper-V](https://www.vmware.com/content/dam/digitalmarketing/vmware/en/pdf/techpaper/vmware-vsphere-cpu-sched-performance-white-paper.pdf). We've already seen RL agents handle generic scheduling and assignment problems and do considerably [better](https://towardsdatascience.com/reinforcement-learning-for-production-scheduling-809db6923419) than conventional optimization techniques, suggesting that we could realize significant gains in many aspects of the system given how ubiquitous the scheduling+assignment problem is in systems. 


## Deployments

How do we deploy a system like this? How can we as humans interpret outcomes? How can we debug these solutions? It's not as straightforward to debug a declarative system. But there are more established methods to productionize AI models like [Claypot](https://www.claypot.ai/) that at least create precedence for how to design these systems, ensure that we can detect model drift (among many other factors) and rollback when the model quality starts slipping.

----------------------------------------------------------------------------------

## Appendix

### Log LLM(?)

I admit that there are components that we can't tune or change a lot, like sharding because it's so expensive to re-shard. And to complement that reduced frequency, does it make sense to rely on historical data? That brings me to log databases - can we query system history to make informed decisions about how to bias the sharding strategy? Can this system history be used to tune primitives and allow them to anticipate Christmas-time load while also learning and handling day-to-day loads? 

What if we could intelligently query system logs so that: 

1. primitives can actively tune themselves based on immediate stressors and *historical data* that's been filtered so only relevant information comes up
2. we can ask where the system failed before and recreate those scenarios to test primitives, i.e. hypothesis testing

Currently, I see a few tools that are capable of doing this with varying levels of success and objectives. [Structured](https://www.ycombinator.com/launches/J9T-structured-analyze-logs-with-ai) is a new company created to address this issue but [entrenched](https://neptune.ai/blog/machine-learning-approach-to-log-analytics) players are also stepping in capture this market. 

### Autoscalers

There are autoscalers that use linear or near-linear models as opposed to very non-linear models and the effect is felt most strongly with [thundering herd issues](https://engineering.fb.com/2015/12/03/ios/under-the-hood-broadcasting-live-video-to-millions/). The problem with using a linear or near-linear model is that for large bursts in traffic, the number of pods doesn't increase fast enough and when it finally does, it's too late since the customers have already experienced significant lag. This happens because if the controller is overdamped, the settling time is too high. And if it's underdamped, we have ringing issues. So we need to tune the controller just right for it to address these traffic bursts at a fast enough rate. But the traffic distribution is constantly changing so we need more complex systems to account for that and change the tuning. My point is that in production, these systems can get very complex and at some point, I would argue for a simpler and more compact RL model to handle the autoscaling task.

Something else I've been trying to argue for, to no avail, is to propagate system signals, e.g. an autoscaler changing the number of pods to X, across the call-path so that downstream services can react appropriately.
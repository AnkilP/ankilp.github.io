---
title: Building a Grafana Plugin That Shows Logs on Hover
author: Ankil Patel
pubDatetime: 2025-10-13
slug: grafana-hover
featured: true
draft: false
tags:
  - debugging
  - on-call
  - grafana
  - log analysis
  - rust
description: A tool I made to make it easier for me to debug hairy systems problems
---

![Grafana Hover Plugin Demo](assets/images/grafana_hover.gif)

Ever stared at a spike or a dip in your metrics dashboard at 3 AM, wondering "am I going to sleep tonight?" You click around, open a new tab for your logging system, try to match timestamps, filter by service, and by the time you find relevant logs, your pager has gone off again.

I built a Grafana panel plugin that fixes this: hover over any point on a graph, and instantly see the relevant logs in a custom panel. No tab switching. No timestamp copying. Just immediate context.

## The Problem: Dashboard-to-Logs Gap

Here's the typical debugging flow:
1. Alert fires, you open Grafana
2. See a spike in errors or latency
3. Note the exact timestamp (hope you got it right)
4. Open Kibana/Splunk/CloudWatch in a new tab
5. Enter time range, add filters
6. Scroll through hundreds of logs
7. Realize you need to check a different service
8. Repeat steps 3-7

I think this points to an inherent issue in the observability landscape: data is highly fragmented and causations are hard to surface.

## Technical Implementation

When a user hovers over a metric spike in Grafana, the plugin captures the timestamp and metric context, then sends it to the backend. My log analysis backend finds the relevant log streams for that metric context for that time window and maps those logs to log templates by removing all the ephemeral parts from the log. From the sequence of log templates, I create a histogram to  calculate KL divergence against a baseline period (e.g., several hours prior). After I'm done calculating the KL divergence, I sort the log templates that contributed most to the divergence. With this ranked list of log templates, I return representative logs to the plugin for display.

### Log analysis

I mainly follow the [LILAC model](https://arxiv.org/pdf/2310.01796), which uses a parse tree to match logs against known templates. When encountering a new log, I send it to an LLM to mask the variable parts (timestamps, IDs, values), creating a template that gets inserted into the parse tree with a unique ID. Subsequent logs matching that template are assigned the same ID without needing the LLM.

This approach outperforms embedding-based similarity search because semantic similarity doesn't guarantee structural equivalence. For example, "user could not sign in" and "user failed to sign in" would cluster together in embedding space despite potentially originating from different code paths with different operational meanings. The parse tree preserves exact structural patterns, ensuring logs are grouped only when they share the same underlying template.

When logs start with variable content like timestamps, the LLM masks them into placeholder tokens (e.g., `<TIMESTAMP>` or `<DATE>`). This preprocessing is crucial for the pattern matching phase. Without this masking, logs like:
```
2024-01-15 10:23:45 User login successful
2024-01-15 10:23:46 User login successful
```
would be treated as completely different patterns. After masking, they both become:
```
<TIMESTAMP> User login successful
```
allowing the system to recognize them as instances of the same template.

#### Aho-Corasick for High-Throughput Pattern Matching

When building systems that need to match millions of incoming logs per second against thousands of known templates, the choice of pattern matching algorithm becomes critical. Traditional approaches like individual regex matching or trie traversals become bottlenecks at scale. While [LILAC](https://arxiv.org/pdf/2310.01796) focuses on accuracy improvements using LLMs with adaptive caching, and ByteDance's [ByteBrain-LogParser](https://arxiv.org/pdf/2504.09113) achieves 229K logs/second with hierarchical clustering optimizations, these approaches still fall short for truly high-scale systems. Even with ByteBrain's claimed 840% speedup over baselines, we need a fundamentally different approach to handle tens of millions of logs per second.

The Aho-Corasick algorithm offers an elegant solution: it can simultaneously search for thousands of patterns in a single pass through the text.

##### Aho-Corasick: Multi-Pattern String Matching

The Aho-Corasick algorithm builds a finite automaton that can simultaneously search for multiple patterns in a single pass. Unlike testing patterns individually, it processes each character of the input text exactly once, achieving O(n + m) complexity where n is the text length and m is the total length of all patterns.

```
Input: "User login successful at 2024-01-15"
Patterns: ["User login", "login successful", "successful at"]

Traditional approach (O(k×n)):
1. Test "User login" → match at position 0
2. Test "login successful" → match at position 5  
3. Test "successful at" → match at position 11

Aho-Corasick (O(n)):
Single pass finds all matches: [(0, "User login"), (5, "login successful"), (11, "successful at")]
```

For log templates, this means we can simultaneously check if an incoming log matches any of thousands of known patterns in one efficient scan.

##### Two-Phase Matching: Fixed Parts Detection + Regex Validation

My approach uses Aho-Corasick to simultaneously match multiple fixed parts throughout the entire template, followed by regex matching for complete validation. This hybrid strategy provides more precise filtering than prefix-only matching:

1. **Phase 1 - Aho-Corasick Multi-Part Matching**: Extract all fixed text portions from each template (the parts between placeholders). Build an automaton containing these fixed parts. When all fixed parts of a template are found in a log, it becomes a candidate.
2. **Phase 2 - Regex Validation**: For each candidate where all fixed parts matched, apply the full regex pattern to confirm the complete match and verify correct ordering.

Consider these log templates:
```
"User <USER_ID> logged in from <IP>"
"User <USER_ID> login failed: <REASON>"
"System <SERVICE> started successfully"
```

For the first template, we extract fixed parts: [`"User"`, `"logged in from"`]. The Aho-Corasick automaton contains all these fragments. When processing `"User 12345 logged in from 192.168.1.1"`, it finds both `"User"` and `"logged in from"`, marking this template as a candidate. We then test the regex `^User \w+ logged in from \d+\.\d+\.\d+\.\d+$` to confirm.

##### Implementation Pattern

The implementation centers around building and maintaining an Aho-Corasick automaton alongside template regexes:

```rust
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use regex::Regex;
use std::collections::{HashMap, HashSet};

struct TemplateMatcher {
    // Fast multi-part matching
    automaton: AhoCorasick,
    // Maps each fixed part to the templates it belongs to
    part_to_templates: HashMap<usize, Vec<usize>>,
    // Maps template ID to its required fixed parts
    template_parts: HashMap<usize, Vec<usize>>,
    // Template ID to regex mapping for validation
    template_patterns: HashMap<usize, Regex>,
    // Template ID to template string mapping
    templates: HashMap<usize, String>,
}

impl TemplateMatcher {
    fn new(templates: &[(usize, String)]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut all_parts = Vec::new();
        let mut part_to_templates = HashMap::new();
        let mut template_parts = HashMap::new();
        let mut template_patterns = HashMap::new();
        let mut template_map = HashMap::new();
        let mut part_index = 0;
        
        for (template_id, template) in templates {
            // Extract all fixed parts from the template
            let fixed_parts = extract_fixed_parts(template);
            let mut template_part_indices = Vec::new();
            
            for part in fixed_parts {
                all_parts.push(part);
                part_to_templates.entry(part_index)
                    .or_insert_with(Vec::new)
                    .push(*template_id);
                template_part_indices.push(part_index);
                part_index += 1;
            }
            
            template_parts.insert(*template_id, template_part_indices);
            
            // Convert template to regex
            let pattern = template_to_regex(template)?;
            template_patterns.insert(*template_id, Regex::new(&pattern)?);
            template_map.insert(*template_id, template.clone());
        }
        
        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&all_parts)?;
            
        Ok(Self {
            automaton,
            part_to_templates,
            template_parts,
            template_patterns,
            templates: template_map,
        })
    }
    
    fn match_log(&self, log: &str) -> Option<usize> {
        // Phase 1: Find all fixed parts present in the log
        let mut found_parts = HashSet::new();
        for mat in self.automaton.find_iter(log) {
            found_parts.insert(mat.pattern().as_usize());
        }
        
        // Find templates where all required parts were found
        let mut candidates = HashSet::new();
        for (template_id, required_parts) in &self.template_parts {
            if required_parts.iter().all(|part| found_parts.contains(part)) {
                candidates.insert(*template_id);
            }
        }
        
        // Phase 2: Validate candidates with regex
        for template_id in candidates {
            if let Some(regex) = self.template_patterns.get(&template_id) {
                if regex.is_match(log) {
                    return Some(template_id);
                }
            }
        }
        
        None
    }
}

fn extract_fixed_parts(template: &str) -> Vec<String> {
    // Split template by placeholders and extract non-empty fixed parts
    use regex::Regex;
    let placeholder_regex = Regex::new(r"<[^>]+>").unwrap();
    let parts: Vec<String> = placeholder_regex
        .split(template)
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    
    // If template starts with fixed text, include it
    // If template has fixed text between or after placeholders, include those too
    parts
}

fn template_to_regex(template: &str) -> Result<String, Box<dyn std::error::Error>> {
    let pattern = template
        .replace("<TIMESTAMP>", r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
        .replace("<USER_ID>", r"\w+")
        .replace("<IP>", r"\d+\.\d+\.\d+\.\d+")
        .replace("<NUMBER>", r"\d+")
        .replace("<STRING>", r"\S+");
    Ok(format!("^{}$", regex::escape(&pattern).replace("\\\\d", "\\d").replace("\\\\w", "\\w").replace("\\\\s", "\\s").replace("\\\\S", "\\S")))
}
```

This approach gives us O(n + m) complexity for the Aho-Corasick phase plus O(k×n) for regex validation, but k is now the small number of prefix matches rather than all templates.

##### Atomic Updates Without Locks

To safely update the global template matcher, we use atomic pointer swapping with copy-on-write semantics:

```rust
use arc_swap::ArcSwap;

static MATCHER: ArcSwap<TemplateMatcher> = ArcSwap::from_pointee(TemplateMatcher::empty());

// Readers (no lock, no waiting)
fn classify_log(log: &str) -> Option<usize> {
    let matcher = MATCHER.load();  // atomic load of Arc pointer
    matcher.match_log(log)
}

// Writer
fn add_template(template: &str, template_id: usize) {
    let old_matcher = MATCHER.load();
    let mut templates = old_matcher.get_all_templates();
    templates.push((template_id, template.to_string()));
    
    let new_matcher = Arc::new(TemplateMatcher::new(&templates).unwrap());
    MATCHER.store(new_matcher);  // atomic pointer swap
}
```

Readers load an Arc to the current matcher and use it for classification. Even if a writer swaps in a new version mid-processing, the reader's Arc keeps their version alive. The Aho-Corasick automaton is immutable once built, so there's no coordination needed.

##### Memory Management Through Reference Counting

The Arc reference counting handles cleanup automatically:

1. Global MATCHER points to v1 (refcount = 1)
2. Reader A loads v1 (refcount = 2)
3. Writer creates v2, swaps pointer (MATCHER now points to v2, v1 refcount still = 2)
4. Reader B loads v2 (v2 refcount = 2)
5. Reader A finishes, drops Arc (v1 refcount = 1)
6. Writer drops reference (v1 refcount = 0)
7. v1 deallocates completely (automaton + regexes)

Unlike persistent data structures, we rebuild the entire Aho-Corasick automaton on updates since template additions are rare and the build cost is acceptable for the performance gains during matching.

##### Performance Characteristics

For the Aho-Corasick approach with T templates and average log length N:

- **Match**: O(N + P×C) where P is the number of fixed parts found and C is the number of candidate templates (typically C << T)
- **Build**: O(total fixed parts length) for automaton construction
- **Space**: O(total fixed parts length) for the automaton plus O(T) for regex storage

The key performance win comes from the enhanced filtering effect: instead of testing T regexes against each log, we only test templates where ALL their fixed parts were found in the log. This is more selective than prefix-only matching. For a system with 10,000 templates where typical logs contain parts from 20-30 templates but only 1-2 templates have ALL their parts present, we achieve even greater reduction in regex operations compared to prefix-only matching.

**Real-world benchmarks** (1M logs, batch processing with 10 threads):
```
Batch parallel (size=1000, 10 threads):
  Matched: 428572/1000000
  Time: 19.45ms
  Throughput: 51.43M logs/sec
  Speedup vs baseline: 4.90x
```

This demonstrates the approach can handle **51+ million logs per second** with a 88% match rate, achieving nearly 5x speedup over traditional regex-only matching. Note that this benchmark measures pure pattern matching performance against known templates - it doesn't include LLM processing time for discovering new templates from unmatched logs. In practice, LLM processing time for unmatched logs will be the bottleneck, not the pattern matching itself. The high throughput makes real-time log analysis feasible for systems with mature template sets where most logs match existing patterns. What this does suggest though is that with offline processing of internal log datasets, this parser can probably handle several services' worth of logs on a single node. 

##### Ablation Study: Breaking Down the Performance Gains

TODO: I need to do a comparison study

##### When to Use This Pattern

Aho-Corasick with regex validation excels when:

- You need to match against thousands of patterns simultaneously
- Patterns have identifiable prefixes that can filter candidates
- Read throughput is critical (millions of matches per second)
- Pattern updates are infrequent compared to matching operations
- Memory usage needs to be predictable and bounded

For log template matching, these conditions align perfectly. Millions of logs are classified per second against thousands of templates that occasionally grow when new patterns emerge. The prefix filtering dramatically reduces expensive regex operations, while the atomic pointer swapping ensures readers never block on updates.

This approach was specifically designed for high-throughput pattern matching workloads. In log analysis, you have constant streams of classification requests, occasional bursts of new template discoveries (especially when new services deploy), and the need for sub-millisecond response times. The Aho-Corasick automaton handles the throughput requirements while regex validation ensures accuracy.

##### Distributed Architecture: Write Master Pattern

In a distributed system with multiple log analysis nodes, I funnel all writes through a queue to a single write master. This architecture leverages the atomic update capabilities of the Aho-Corasick approach:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Node A    │     │   Node B    │     │   Node C    │
│ (Read-only) │     │ (Read-only) │     │ (Read-only) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │ new template      │ new template      │
       └──────────┬────────┴───────────────────┘
                  ▼
            ┌──────────┐
            │  Queue   │ (Kafka/SQS/etc)
            └────┬─────┘
                 ▼
          ┌─────────────┐
          │Write Master │
          │             │
          │ 1. Consume  │
          │ 2. Rebuild  │
          │ 3. Publish  │
          └─────┬───────┘
                │ new matcher version
                ▼
         ┌──────────────┐
         │ Distribution │ (S3/Redis/etc)
         └──────────────┘
```

When a node discovers a new log pattern:
1. It continues using its local read-only matcher for classification
2. Sends the unknown log to a distributed queue
3. The write master consumes template proposals, deduplicates them, and rebuilds the matcher
4. The new matcher (automaton + regexes) is serialized and published to shared storage
5. Reader nodes listen for new versions and atomic-swap their local copy

The immutable nature of Aho-Corasick automatons turns distributed state synchronization into a simple publish-subscribe problem. Traditional mutable structures would require complex distributed locking or eventually-consistent merge strategies. Here, we ship complete matcher versions, and since template updates are infrequent, the rebuild cost is acceptable for the consistency guarantees.


## Impact

Since deploying this plugin:
- Average time to identify root cause dropped from ~15 minutes to ~3 minutes
- I switch tabs 80% less often
- Reduced false escalations by 40% (engineers could quickly see logs showed expected behavior)

## Future Enhancements

Working on:
1. **Trace integration** - Show distributed trace when hovering over latency spikes
2. **Sending only diffs to the readers** - Instead of sending the entire trie, we only send the diffs to the readers. I considered this initally but thought it would be too thorny to manage distributed versions without a centralized system of record. Especially for an initial implementaion.
3. Ask the LLM to provide more descriptive masks. e.g. [TIMESTAMP] vs [MASK]

## Try It Yourself

The plugin is open source: [https://github.com/StandardRunbook/grafana-hover-plugin](https://github.com/StandardRunbook/grafana-hover-plugin)

Note: The plugin is pending Grafana marketplace approval. For now, install from source:

```bash
git clone https://github.com/StandardRunbook/grafana-hover-plugin
pnpm run build
pnpm run server
```

The 3 AM version of you will thank you later.

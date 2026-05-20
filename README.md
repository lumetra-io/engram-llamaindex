# llama-index-memory-engram

Durable, explainable memory for LlamaIndex agents — powered by [Engram](https://lumetra.io).

`EngramMemory` is a `BaseMemory` implementation that replaces LlamaIndex's
built-in chat-history buffer with Engram's hybrid retrieval pipeline
(BM25 + vector + knowledge graph + reranker). Every message your agent
sees is persisted to an Engram bucket; reads come back in chronological
order, and semantic recall is one call away.

## Install

```bash
pip install llama-index-memory-engram
```

## Usage

```python
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.memory.engram import EngramMemory

os.environ["ENGRAM_API_KEY"] = "eng_live_..."   # or pass api_key=... explicitly

memory = EngramMemory.from_defaults(
    bucket="user-42",         # one bucket per user / session / agent
    read_limit=50,            # how many recent messages get() returns
)

agent = FunctionAgent(
    llm=OpenAI("gpt-4o"),
    tools=[...],
)

response = await agent.run(
    "What did we decide about the Q3 launch?",
    memory=memory,
)
```

Get an API key at <https://lumetra.io>. Keys look like `eng_live_...`.

### Direct semantic recall

`get()` returns the most recent `read_limit` messages, which is what
agents expect from chat history. When you want hybrid retrieval over the
*entire* bucket, call `query()` directly:

```python
result = memory.query("regulatory risks we discussed last quarter")
print(result["answer"])
print(result["memories_found"])
```

### Bucket scoping

Pick a bucket name per logical conversation scope:

```python
EngramMemory(bucket=f"user-{user_id}")           # per user
EngramMemory(bucket=f"session-{session_id}")     # per session
EngramMemory(bucket=f"agent-{agent_id}")         # per agent
```

Buckets are created on first write — no admin call needed.

### Self-hosted Engram

```python
EngramMemory(
    bucket="ops",
    base_url="https://engram.internal.example.com",
    api_key="...",
)
```

## API reference

| Method | Behavior |
|---|---|
| `put(message)` | Append one `ChatMessage` to the bucket. |
| `put_messages(messages)` | Append many. |
| `get(input=None)` | Return the most recent `read_limit` messages, oldest-first. |
| `get_all()` | Same as `get()`. |
| `set(messages)` | Clear the bucket, then write `messages`. |
| `reset()` | Clear the bucket. |
| `query(question)` | Hybrid retrieval over the entire bucket. Returns the raw Engram response. |
| `list_buckets(limit, offset)` | List buckets visible to this API key. |
| `delete_memory(memory_id)` | Delete a single memory by id. |

All methods have async equivalents (`aput`, `aget`, ...) inherited from
`BaseMemory`; they currently run the sync implementation in a thread.

## Configuration

| Constructor arg | Env var | Default |
|---|---|---|
| `api_key` | `ENGRAM_API_KEY` | _required_ |
| `bucket` | — | `"default"` |
| `base_url` | — | `"https://api.lumetra.io"` |
| `read_limit` | — | `50` |
| `timeout` | — | `120.0` |

## License

MIT — see [LICENSE](./LICENSE).

For data-handling details see [PRIVACY.md](./PRIVACY.md) and
<https://lumetra.io/privacy>.

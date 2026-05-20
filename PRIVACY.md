# Privacy

This package sends the chat messages your LlamaIndex agent passes to its
memory methods — message content, the optional `input` query, and the
configured `bucket` — to the Engram REST API at `https://api.lumetra.io`
(or the self-hosted base URL you configured). Memories are stored under
your Engram tenant, scoped by the API key you provided via constructor
or the `ENGRAM_API_KEY` environment variable.

The package does not collect, log, or transmit data to any third party
other than the Engram service you've explicitly authorized. It does not
read other LlamaIndex resources (indexes, documents, tools) — only the
chat messages and queries supplied to each memory call.

For Engram's own data-handling and retention policy, see
<https://lumetra.io/privacy>.

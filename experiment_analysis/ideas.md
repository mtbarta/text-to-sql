public benchmarks; https://llm-benchmark.tinybird.live/

we have a few areas of investment:
- the prompt. how do we ensure that the agent knows how to accomplish the task.
- the model. "openai/gpt-oss-120b:nitro" is not the best anymore, let's upgrade to a better model.
- retrieval.
    - how can we improve precision/recall of the rules we retrieve?


easy baseline - run 20260423-02111 -- 70%
- push the model to iterate more.
- validate the result more.
- prevent some generation failures.

hard baseline - run 20260423_030920 -- 42%
- filtering and group by failures

hard #2 -- run_20260423_053812 -- 50%
intuition; it's possible a better model will follow the prompt better, thereby leading to
better sql validation of filters.
- try kimi 2.6 model as a better sql benchmark.
- lower temperature to prevent spurious filters.

result: no more single query trajectories, but similar amount of wrong filter.


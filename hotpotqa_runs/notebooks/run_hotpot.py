# %% [markdown]
# #### Notebook for running Chain-of-Thought with no supporting context experiments

# %%
import sys, os
sys.path.append('/Users/xinzheli/git_repo/reflexion/hotpotqa_runs/')
root = '../root/'
from llm import AnyOpenAILLM
model_name = 'gpt-3.5-turbo'
deployment_id = None
self_reflect_llm = AnyOpenAILLM(
                        temperature=0,
                        max_tokens=250,
                        model_name=model_name,
                        deployment_id=deployment_id,
                        model_kwargs={"stop": "\n"},
                        openai_api_key=os.environ['OPENAI_API_KEY'])

# %%
from util import summarize_trial, log_trial, save_agents
import joblib
from agents import CoTAgent, ReflexionStrategy

# %% [markdown]
# #### Load the HotPotQA Sample

# %%
hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

# %% [markdown]
# #### Define the Reflexion Strategy

# %%
print(ReflexionStrategy.__doc__)

# %%
strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION

# %% [markdown]
# #### Initialize a CoTAgent for each question

# %%
from prompts import cot_simple_reflect_agent_prompt, cot_simple_reflect_prompt, cot_simple_agent_prompt
from fewshots import COTQA_SIMPLE6, COT_SIMPLE_REFLECTION


# %%
row = next(hotpot.iterrows())[1]
COTQA_SIMPLE6.split('\n')

# %%
COT_SIMPLE_REFLECTION.split('\n')

# %%

agents = [CoTAgent(question = row['question'],
                   context = '',
                   key = row['answer'],
                   agent_prompt=cot_simple_agent_prompt if strategy == ReflexionStrategy.NONE else cot_simple_reflect_agent_prompt,
                   cot_examples = COTQA_SIMPLE6,
                   reflect_prompt = cot_simple_reflect_prompt,
                   reflect_examples = COT_SIMPLE_REFLECTION,
                   self_reflect_llm=self_reflect_llm
                      ) ]

# %%
strategy

# %% [markdown]
# #### Run `n` trials

# %%
n = 2
trial = 0
log = ''
for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        agent.run(reflexion_strategy = strategy)
        print(f'Answer: {agent.key}')
        
    trial += 1
    log += log_trial(agents, trial)
    correct, incorrect = summarize_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

# %% [markdown]
# #### Save the result log

# %%
with open(os.path.join(root, 'CoT', 'no_context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join(root, 'CoT', 'no_context', strategy.value, 'agents'))



import sys, os
sys.path.append('/Users/xinzheli/git_repo/reflexion/hotpotqa_runs/')
root = '../root/'
from llm import AnyOpenAILLM, chatgpt_llm
from util import summarize_trial, log_trial, save_agents
import joblib
from agents import CoTAgent, ReflexionStrategy
                    
hotpot = joblib.load('hotpotqa_runs/data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
# print(ReflexionStrategy.__doc__)
    # NONE: No reflection
    # LAST_ATTEMPT: Use last reasoning trace in context 
    # REFLEXION: Apply reflexion to the next reasoning trace 
    # LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION

from prompts import cot_simple_reflect_agent_prompt, cot_simple_reflect_prompt, cot_simple_agent_prompt
from fewshots import COTQA_SIMPLE6, COT_SIMPLE_REFLECTION
row = next(hotpot.iterrows())[1]
COTQA_SIMPLE6.split('\n')
COT_SIMPLE_REFLECTION.split('\n')

agents = [CoTAgent(question = row['question'],
                   context = '',
                   key = row['answer'],
                   agent_prompt=cot_simple_agent_prompt if strategy == ReflexionStrategy.NONE else cot_simple_reflect_agent_prompt,
                   cot_examples = COTQA_SIMPLE6,
                   reflect_prompt = cot_simple_reflect_prompt,
                   reflect_examples = COT_SIMPLE_REFLECTION,
                   self_reflect_llm=chatgpt_llm,
                   action_llm=chatgpt_llm) ]
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

with open(os.path.join(root, 'CoT', 'no_context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join(root, 'CoT', 'no_context', strategy.value, 'agents'))



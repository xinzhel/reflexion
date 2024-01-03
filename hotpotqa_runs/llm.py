import os
import openai
import pandas as pd
import json

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get('model_name', 'gpt-3.5-turbo') 
        self.deployment_id = kwargs.get('deployment_id', None) 
        self.gpt_usage_record = GPTUsageRecord()
        self.model_type = 'chat'

        self.id = 1
        
    def __call__(self, usr_msg, system_msg=''):
        if self.id == 1:
            self.id += 1
            return "Let's think step by step. VIVA Media AG changed its name in 2004. The new acronym must stand for the new name of the company. Unfortunately, without further information, it is not possible to determine the new acronym."
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_msg}, 
                      {"role": "user", "content": usr_msg} ] ,
            temperature=0.3, # 0.9 more random
            # max_tokens=150,
            top_p=1.0,  # top_p should be a float between 0 and 1
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        # update the usage record
        for key in self.gpt_usage_record.episode_usage.keys():
            self.gpt_usage_record.episode_usage[key] += completion.usage.to_dict()[key]
        
        return completion.choices[0].message['content']
        
class GPTUsageRecord:
    def __init__(self):
        self.episode_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # update the total usage of the model
    def write_usage(self):
        file = model+'_usage.json'
        # if exist
        if os.path.exists(file):
            with open(file, 'r') as f:
                previous_usage = json.load(f)
        else:
            previous_usage = {'total': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 'recent': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}


        # write gpt usage record to json
        for key in previous_usage['total'].keys():
            previous_usage['total'][key] += self.episode_usage[key]

        for key in previous_usage['recent'].keys():
            previous_usage['recent'][key] = self.episode_usage[key]


        with open(model+'_usage.json', 'w') as f:
            json.dump(previous_usage, f)
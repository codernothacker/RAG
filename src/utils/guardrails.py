from typing import Optional, List, Dict
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class EnhancedGuardrails:
    def __init__(self, llm):
        self.llm = llm
        self.content_policies = {
            'context_relevance': {
                'enabled': True,
                'threshold': 0.7
            },
            'response_length': {
                'enabled': True,
                'min_length': 10,
                'max_length': 2000
            },
            'topic_boundaries': {
                'enabled': True,
                'blocked_topics': [
                    'harmful_instructions',
                    'explicit_content',
                    'hate_speech',
                    'personal_information',
                    'financial_advice',
                    'medical_advice'
                ]
            }
        }
        self.eval_prompt = PromptTemplate(
            input_variables=["response", "context", "query"],
            template="""Please evaluate this response based on the following criteria:
            1. Is it relevant to the provided context? (Score 0-1)
            2. Does it contain any harmful or inappropriate content? (Yes/No)
            3. Does it stay within the topic boundaries? (Yes/No)
            4. Is it factually consistent with the context? (Yes/No)

            Context: {context}
            Query: {query}
            Response: {response}

            Provide your evaluation in JSON format:
            {{"relevance_score": float, "contains_harmful": bool, "within_boundaries": bool, "factually_consistent": bool}}
            """
        )
        
        self.eval_chain = LLMChain(llm=self.llm, prompt=self.eval_prompt)

    def enforce_context_relevance(self, response: str, context: str, query: str) -> bool:
        """Check if response is relevant to the context"""
        if not self.content_policies['context_relevance']['enabled']:
            return True
            
        try:
            eval_result = self.eval_chain.run(
                response=response,
                context=context,
                query=query
            )
            import json
            result = json.loads(eval_result)
            
            return result['relevance_score'] >= self.content_policies['context_relevance']['threshold']
        except Exception as e:
            print(f"Error in relevance check: {e}")
            return True

    def check_response_length(self, response: str) -> bool:
        """Verify response length is within acceptable bounds"""
        if not self.content_policies['response_length']['enabled']:
            return True
            
        length = len(response.split())
        return (length >= self.content_policies['response_length']['min_length'] and 
                length <= self.content_policies['response_length']['max_length'])

    def check_topic_boundaries(self, response: str) -> Dict[str, bool]:
        """Check for forbidden topics and content"""
        if not self.content_policies['topic_boundaries']['enabled']:
            return {'safe': True, 'violations': []}
            
        violations = []
        topic_patterns = {
            'harmful_instructions': r'\b(hack|exploit|attack|break into|steal)\b',
            'explicit_content': r'\b(explicit|nsfw|adult|xxx)\b',
            'hate_speech': r'\b(hate|slur|discriminat|racial)\b',
            'personal_information': r'\b(password|credit card|social security|address)\b',
            'financial_advice': r'\b(invest|stock|trade|buy|sell)\b(?=.*\b(recommend|should|advise)\b)',
            'medical_advice': r'\b(diagnose|treat|cure|healing|medicine)\b(?=.*\b(should|recommend|advise)\b)'
        }
        
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, response.lower()):
                violations.append(topic)
        
        return {
            'safe': len(violations) == 0,
            'violations': violations
        }

    def filter_response(self, response: str, context: str, query: str) -> Optional[str]:
        """Apply all guardrails and filter response"""
        try:
            if not self.check_response_length(response):
                return self._get_fallback_response("Response length outside acceptable bounds")
            if not self.enforce_context_relevance(response, context, query):
                return self._get_fallback_response("Response not relevant to context")
            topic_check = self.check_topic_boundaries(response)
            if not topic_check['safe']:
                return self._get_fallback_response(
                    f"Response contains inappropriate content: {', '.join(topic_check['violations'])}"
                )
            return response
            
        except Exception as e:
            print(f"Error in guardrails: {e}")
            return self._get_fallback_response("Error in response filtering")

    def _get_fallback_response(self, reason: str) -> str:
        """Generate appropriate fallback response"""
        fallback_template = """I apologize, but I need to stay focused on providing helpful information 
        that is directly relevant to the documents and context provided. Could you please rephrase your 
        question or specify what information you're looking for from the uploaded documents?"""
        
        return fallback_template
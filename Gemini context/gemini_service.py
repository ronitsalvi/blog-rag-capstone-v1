"""
Gemini AI Service for generating workflow JSON structures
Integrates with Google Gemini 2.5-flash model for natural language processing
"""

import json
import logging
from typing import Dict, Any
import requests
from config import CONFIG

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google Gemini AI API"""
    
    def __init__(self):
        self.api_key = CONFIG.GEMINI['API_KEY']
        self.model = CONFIG.GEMINI['MODEL']
        self.base_url = CONFIG.GEMINI['BASE_URL']
        logger.info(f"Initialized Gemini service with model: {self.model}")
    
    async def generate_workflow_json(self, user_requirement: str) -> Dict[str, Any]:
        """
        Generate workflow JSON structure from user requirement
        Uses Gemini AI to convert natural language to structured workflow data
        """
        try:
            logger.info(f"Generating workflow JSON for requirement: {user_requirement[:100]}...")
            
            # Build prompt for Gemini
            prompt = self._build_comprehensive_prompt(user_requirement)
            
            # Call Gemini API
            response_text = await self._call_gemini_api(prompt)
            
            # Parse response
            result = self._parse_gemini_response(response_text)
            
            logger.info("Successfully generated workflow JSON from Gemini")
            return {
                "success": True,
                "data": result,
                "raw_response": response_text
            }
            
        except Exception as e:
            logger.error(f"Failed to generate workflow JSON: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": self._get_fallback_structure()
            }
    
    def _build_comprehensive_prompt(self, user_requirement: str) -> str:
        """Build generic workflow generation prompt for Gemini AI"""
        return """You are a workflow generation AI that creates JSON structures for document signing workflows.

IMPORTANT: Generate ONLY valid JSON in your response. No markdown blocks, no explanations, no additional text.

User Requirement: """ + user_requirement + """

Create a JSON structure with dynamicProperties and workflowData sections based on the user's requirement."""
    
    def _parse_gemini_response(self, text: str) -> Dict[str, Any]:
        """Parse Gemini AI response text into JSON structure"""
        try:
            # Remove any markdown formatting
            text = text.strip()
            if text.startswith('```json'):
                text = text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {str(e)}")
            logger.debug(f"Raw response: {text}")
            
            # Return minimal fallback structure
            return {
                "dynamicProperties": {
                    "documents#p": {
                        "attributeRef": "documents",
                        "v": True,
                        "e": True,
                        "m": False,
                        "type": "TAB_LIST"
                    }
                },
                "workflowData": {
                    "documents": [
                        {
                            "id": "document1",
                            "documentName": "Document 1",
                            "subDocuments": [],
                            "stamps": {
                                "mergedDocumentStamp": True,
                                "stampSeries": {
                                    "stampSeriesEnabled": False,
                                    "seriesConfig": [{}]
                                }
                            }
                        }
                    ]
                }
            }
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API with the given prompt"""
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 0.8,
                "maxOutputTokens": 8192,
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract text from Gemini response
            if 'candidates' in response_data and response_data['candidates']:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    return candidate['content']['parts'][0]['text']
            
            raise Exception("No valid response from Gemini API")
            
        except requests.RequestException as e:
            logger.error(f"Gemini API request failed: {str(e)}")
            raise Exception(f"Gemini API call failed: {str(e)}")
        except Exception as e:
            logger.error(f"Gemini API response parsing failed: {str(e)}")
            raise Exception(f"Failed to parse Gemini response: {str(e)}")
    
    def _get_fallback_structure(self) -> Dict[str, Any]:
        """Return fallback JSON structure when Gemini fails"""
        return {
            "dynamicProperties": {
                "documents#p": {
                    "attributeRef": "documents",
                    "v": True,
                    "e": True,
                    "m": False,
                    "type": "TAB_LIST"
                }
            },
            "workflowData": {
                "documents": [
                    {
                        "id": "document1",
                        "documentName": "Fallback Document",
                        "subDocuments": [],
                        "stamps": {
                            "mergedDocumentStamp": True,
                            "stampSeries": {
                                "stampSeriesEnabled": False,
                                "seriesConfig": [{}]
                            }
                        }
                    }
                ]
            }
        }
import os
import json
from typing import Dict, List, Tuple, Optional, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import (
    SystemMessage,
    HumanMessage, 
    AIMessage,
    BaseMessage
)
import chromadb
from chromadb.utils import embedding_functions
import ollama
from langchain.graphs import Neo4jGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Constants
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """You are a knowledge graph builder specialized in extracting structured information from text.

Analyze the following text from a video transcript and extract key entities and their relationships.
Focus on the main concepts, people, organizations, products, and other important entities.

For each entity, provide:
1. An entity ID (lowercase, underscore-separated)
2. The entity name (as mentioned in the text)
3. Entity type (PERSON, CONCEPT, ORGANIZATION, PRODUCT, LOCATION, etc.)
4. A brief description based on information in the text

For relationships between entities, provide:
1. Source entity ID
2. Target entity ID 
3. Relationship type (explains, works_for, created_by, part_of, etc.)
4. A brief description of the relationship

Only extract information explicitly stated in the text. Do not infer relationships unless clearly implied.

TEXT TO ANALYZE:
{text}

EXPECTED OUTPUT FORMAT:
```json
{
  "entities": [
    {
      "id": "entity_id",
      "name": "Entity Name",
      "type": "ENTITY_TYPE",
      "description": "Description from text"
    }
  ],
  "relationships": [
    {
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "description": "Description of relationship"
    }
  ]
}
```

Provide only valid JSON as output with no other text.
"""

# Knowledge verification prompt
KNOWLEDGE_VERIFICATION_PROMPT = """You are a critical information validator for knowledge graph construction.

Review the following extracted entities and relationships from a video transcript:

{extracted_data}

Your task is to verify the accuracy and quality of these extractions by checking:
1. Are all entities relevant to the main topics in the text?
2. Are the relationships logically valid and supported by the text?
3. Are there any contradictions or inconsistencies?
4. Are entity types correctly assigned?

ORIGINAL TEXT SEGMENT:
{text}

Provide your assessment in the following format:
```json
{
  "is_valid": true/false,
  "entities_to_remove": ["entity_id1", "entity_id2"],
  "relationships_to_remove": [{"source": "entity_id1", "target": "entity_id2", "type": "TYPE"}],
  "suggestions": ["suggestion1", "suggestion2"]
}
```

Only provide JSON output with no additional text.
"""

# Enhanced answer generation prompt
ENHANCED_ANSWER_PROMPT = """You are a highly knowledgeable AI assistant that uses both structured knowledge and context to provide accurate answers.

CONTEXT INFORMATION:
{context}

KNOWLEDGE GRAPH ENTITIES:
{entities}

KNOWLEDGE GRAPH RELATIONSHIPS:
{relationships}

QUESTION: {question}

When answering:
1. Use the context information from the transcript segments as your primary source
2. Use the knowledge graph entities and relationships to provide structured understanding
3. If the information isn't in the sources, say "This information is not in the provided sources."
4. When using information from the knowledge graph, mention it explicitly
5. Keep your answer concise but complete

ANSWER:
"""

class KnowledgeGraphBuilder:
    def __init__(
        self, 
        model_name: str = OLLAMA_MODEL, 
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        self.model_name = model_name
        
        # Initialize Neo4j if credentials are provided
        self.use_neo4j = all([neo4j_uri, neo4j_username, neo4j_password])
        if self.use_neo4j:
            try:
                self.graph = Neo4jGraph(
                    url=neo4j_uri,
                    username=neo4j_username,
                    password=neo4j_password
                )
                print("Neo4j connection established")
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j: {e}")
                print("Will store graph data in memory")
                self.use_neo4j = False
                self.in_memory_graph = {"entities": {}, "relationships": []}
        else:
            print("Neo4j credentials not provided. Using in-memory graph.")
            self.in_memory_graph = {"entities": {}, "relationships": []}
            
        # Setup state graph for the LangGraph workflow
        self.graph_workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for entity extraction and verification."""
        # Define nodes (workflow steps)
        workflow = StateGraph({"messages": [], "extracted_data": None, "verified_data": None})
        
        # Add nodes
        workflow.add_node("entity_extraction", self._extract_entities)
        workflow.add_node("entity_verification", self._verify_entities)
        workflow.add_node("graph_storage", self._store_in_graph)
        
        # Define edges
        workflow.add_edge("entity_extraction", "entity_verification")
        workflow.add_edge("entity_verification", "graph_storage")
        workflow.add_edge("graph_storage", END)
        
        # Set entry point
        workflow.set_entry_point("entity_extraction")
        
        return workflow.compile()
    
    def _extract_entities(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities and relationships from text."""
        messages = state["messages"]
        text = messages[-1].content if messages else ""
        
        if not text:
            return {"messages": messages, "extracted_data": {"entities": [], "relationships": []}}
        
        try:
            # Call Ollama API for entity extraction
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(text=text)}]
            )
            
            response_content = response.message.content if hasattr(response.message, "content") else ""
            
            # Extract JSON from response
            json_start = response_content.find('```json')
            json_end = response_content.rfind('```')
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_content[json_start + 7:json_end].strip()
            else:
                json_text = response_content
            
            # Parse extracted data
            extracted_data = json.loads(json_text)
            
            # Update state
            new_state = state.copy()
            new_state["extracted_data"] = extracted_data
            
            # Add AI message
            ai_message = AIMessage(content=f"Extracted {len(extracted_data.get('entities', []))} entities and {len(extracted_data.get('relationships', []))} relationships.")
            new_state["messages"] = add_messages(state["messages"], [ai_message])
            
            return new_state
            
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return {"messages": messages, "extracted_data": {"entities": [], "relationships": []}}
    
    def _verify_entities(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify and clean up extracted entities and relationships."""
        messages = state["messages"]
        extracted_data = state.get("extracted_data", {"entities": [], "relationships": []})
        
        if not extracted_data or not extracted_data.get("entities"):
            return {"messages": messages, "extracted_data": extracted_data, "verified_data": extracted_data}
        
        original_text = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                original_text = message.content
                break
        
        try:
            # Call Ollama API for verification
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user", 
                    "content": KNOWLEDGE_VERIFICATION_PROMPT.format(
                        extracted_data=json.dumps(extracted_data, indent=2),
                        text=original_text
                    )
                }]
            )
            
            response_content = response.message.content if hasattr(response.message, "content") else ""
            
            # Extract JSON from response
            json_start = response_content.find('```json')
            json_end = response_content.rfind('```')
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_content[json_start + 7:json_end].strip()
            else:
                json_text = response_content
            
            # Parse verification result
            verification_result = json.loads(json_text)
            
            # Filter out entities and relationships based on verification
            verified_data = extracted_data.copy()
            
            if verification_result.get("is_valid") is False:
                # Remove invalid entities
                entities_to_remove = verification_result.get("entities_to_remove", [])
                verified_data["entities"] = [
                    e for e in verified_data["entities"] 
                    if e.get("id") not in entities_to_remove
                ]
                
                # Remove invalid relationships
                relationships_to_remove = verification_result.get("relationships_to_remove", [])
                verified_data["relationships"] = [
                    r for r in verified_data["relationships"]
                    if not any(
                        r.get("source") == rr.get("source") and 
                        r.get("target") == rr.get("target") and
                        r.get("type") == rr.get("type")
                        for rr in relationships_to_remove
                    )
                ]
            
            # Update state
            new_state = state.copy()
            new_state["verified_data"] = verified_data
            
            # Add AI message
            ai_message = AIMessage(content=f"Verified data: {len(verified_data.get('entities', []))} entities and {len(verified_data.get('relationships', []))} relationships.")
            new_state["messages"] = add_messages(state["messages"], [ai_message])
            
            return new_state
            
        except Exception as e:
            print(f"Error in entity verification: {e}")
            return {"messages": messages, "extracted_data": extracted_data, "verified_data": extracted_data}
    
    def _store_in_graph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Store verified entities and relationships in the knowledge graph."""
        verified_data = state.get("verified_data", {"entities": [], "relationships": []})
        
        if not verified_data:
            return state
        
        try:
            if self.use_neo4j:
                # Store in Neo4j
                self._store_in_neo4j(verified_data)
            else:
                # Store in memory
                self._store_in_memory(verified_data)
            
            # Add AI message
            ai_message = AIMessage(content=f"Successfully stored {len(verified_data.get('entities', []))} entities and {len(verified_data.get('relationships', []))} relationships in the knowledge graph.")
            new_state = state.copy()
            new_state["messages"] = add_messages(state["messages"], [ai_message])
            
            return new_state
            
        except Exception as e:
            print(f"Error storing in graph: {e}")
            ai_message = AIMessage(content=f"Error storing in knowledge graph: {str(e)}")
            new_state = state.copy()
            new_state["messages"] = add_messages(state["messages"], [ai_message])
            return new_state
    
    def _store_in_neo4j(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Store verified data in Neo4j."""
        # Create entities
        for entity in data.get("entities", []):
            # Create Cypher query for entity
            properties = {
                "name": entity.get("name", ""),
                "description": entity.get("description", "")
            }
            
            query = (
                f"MERGE (e:{entity.get('type')} {{id: $id}}) "
                f"SET e += $properties "
                f"RETURN e"
            )
            
            self.graph.query(
                query,
                {"id": entity.get("id"), "properties": properties}
            )
        
        # Create relationships
        for rel in data.get("relationships", []):
            # Create Cypher query for relationship
            query = (
                f"MATCH (source {{id: $source_id}}), (target {{id: $target_id}}) "
                f"MERGE (source)-[r:{rel.get('type')} {{description: $description}}]->(target) "
                f"RETURN r"
            )
            
            self.graph.query(
                query,
                {
                    "source_id": rel.get("source"),
                    "target_id": rel.get("target"),
                    "description": rel.get("description", "")
                }
            )
    
    def _store_in_memory(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Store verified data in memory."""
        # Store entities
        for entity in data.get("entities", []):
            entity_id = entity.get("id")
            if entity_id:
                self.in_memory_graph["entities"][entity_id] = entity
        
        # Store relationships
        for rel in data.get("relationships", []):
            self.in_memory_graph["relationships"].append(rel)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text to extract entities and relationships."""
        # Split text into chunks if it's too long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200
        )
        
        # Initialize result containers
        all_entities = []
        all_relationships = []
        
        if len(text) > 10000:
            # Process in chunks
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                # Initialize with human message (the chunk)
                initial_state = {"messages": [HumanMessage(content=chunk)], "extracted_data": None, "verified_data": None}
                
                # Execute workflow
                final_state = self.graph_workflow.invoke(initial_state)
                
                # Collect verified data
                verified_data = final_state.get("verified_data", {})
                all_entities.extend(verified_data.get("entities", []))
                all_relationships.extend(verified_data.get("relationships", []))
        else:
            # Process entire text
            initial_state = {"messages": [HumanMessage(content=text)], "extracted_data": None, "verified_data": None}
            final_state = self.graph_workflow.invoke(initial_state)
            verified_data = final_state.get("verified_data", {})
            all_entities = verified_data.get("entities", [])
            all_relationships = verified_data.get("relationships", [])
        
        # Deduplicate entities by ID
        unique_entities = {}
        for entity in all_entities:
            entity_id = entity.get("id")
            if entity_id and entity_id not in unique_entities:
                unique_entities[entity_id] = entity
        
        # Deduplicate relationships
        unique_relationships = []
        relationship_keys = set()
        for rel in all_relationships:
            key = f"{rel.get('source')}|{rel.get('target')}|{rel.get('type')}"
            if key not in relationship_keys:
                relationship_keys.add(key)
                unique_relationships.append(rel)
        
        return {
            "entities": list(unique_entities.values()),
            "relationships": unique_relationships
        }
    
    def get_relevant_subgraph(self, query: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a relevant subgraph for a given query."""
        if self.use_neo4j:
            return self._get_neo4j_subgraph(query, video_id)
        else:
            return self._get_memory_subgraph(query)
    
    def _get_neo4j_subgraph(self, query: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a relevant subgraph from Neo4j based on query."""
        # Convert query to keywords
        try:
            # Get keywords from query using Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user", 
                    "content": f"Extract 3-5 key entities or concepts from this question. Output just the comma-separated list of key terms, no other text:\n\n{query}"
                }]
            )
            
            keywords = response.message.content
            # Clean up keywords
            keywords = keywords.replace(".", "").replace("\n", "").strip()
            keywords_list = [k.strip() for k in keywords.split(",")]
            
            # Create Cypher query to find relevant subgraph
            cypher_query = """
            MATCH (e)
            WHERE e.name CONTAINS $keyword1 OR e.description CONTAINS $keyword1
            OR e.name CONTAINS $keyword2 OR e.description CONTAINS $keyword2
            WITH e
            MATCH (e)-[r]-(related)
            RETURN e, r, related
            LIMIT 20
            """
            
            params = {
                "keyword1": keywords_list[0] if keywords_list else "",
                "keyword2": keywords_list[1] if len(keywords_list) > 1 else keywords_list[0] if keywords_list else ""
            }
            
            result = self.graph.query(cypher_query, params)
            
            # Format result for frontend
            entities = []
            relationships = []
            
            # Process nodes
            nodes_seen = set()
            for row in result:
                for node_key in ["e", "related"]:
                    if node_key in row and row[node_key]["id"] not in nodes_seen:
                        nodes_seen.add(row[node_key]["id"])
                        entities.append({
                            "id": row[node_key]["id"],
                            "name": row[node_key].get("name", ""),
                            "type": next((label for label in row[node_key].labels if label != "Entity"), "Entity"),
                            "description": row[node_key].get("description", "")
                        })
                
                # Process relationships
                if "r" in row:
                    rel_type = row["r"].type
                    source_id = row["e"]["id"]
                    target_id = row["related"]["id"]
                    relationships.append({
                        "source": source_id,
                        "target": target_id,
                        "type": rel_type,
                        "description": row["r"].get("description", "")
                    })
            
            return {
                "entities": entities,
                "relationships": relationships
            }
            
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
            return {"entities": [], "relationships": []}
    
    def _get_memory_subgraph(self, query: str) -> Dict[str, Any]:
        """Get a relevant subgraph from in-memory storage based on query."""
        try:
            # Get keywords from query using Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user", 
                    "content": f"Extract 3-5 key entities or concepts from this question. Output just the comma-separated list of key terms, no other text:\n\n{query}"
                }]
            )
            
            keywords = response.message.content
            # Clean up keywords
            keywords = keywords.replace(".", "").replace("\n", "").strip()
            keywords_list = [k.strip().lower() for k in keywords.split(",")]
            
            # Filter entities based on keywords
            relevant_entity_ids = set()
            for entity_id, entity in self.in_memory_graph["entities"].items():
                entity_name = entity.get("name", "").lower()
                entity_desc = entity.get("description", "").lower()
                
                if any(kw in entity_name or kw in entity_desc for kw in keywords_list):
                    relevant_entity_ids.add(entity_id)
            
            # Add connected entities
            connected_entity_ids = set(relevant_entity_ids)
            for rel in self.in_memory_graph["relationships"]:
                source = rel.get("source")
                target = rel.get("target")
                
                if source in relevant_entity_ids:
                    connected_entity_ids.add(target)
                if target in relevant_entity_ids:
                    connected_entity_ids.add(source)
            
            # Get relevant entities and relationships
            relevant_entities = [
                self.in_memory_graph["entities"][entity_id]
                for entity_id in connected_entity_ids
                if entity_id in self.in_memory_graph["entities"]
            ]
            
            relevant_relationships = [
                rel for rel in self.in_memory_graph["relationships"]
                if rel.get("source") in connected_entity_ids and rel.get("target") in connected_entity_ids
            ]
            
            return {
                "entities": relevant_entities,
                "relationships": relevant_relationships
            }
            
        except Exception as e:
            print(f"Error getting in-memory subgraph: {e}")
            return {"entities": [], "relationships": []}
    
    def get_enhanced_answer(self, question: str, context: str, video_id: Optional[str] = None) -> str:
        """Generate an enhanced answer using both context and knowledge graph."""
        # Get relevant subgraph
        relevant_graph = self.get_relevant_subgraph(question, video_id)
        
        # Format entities and relationships for prompt
        entities_text = "\n".join([
            f"- {e.get('name')} ({e.get('type')}): {e.get('description', 'No description')}"
            for e in relevant_graph.get("entities", [])
        ])
        
        relationships_text = "\n".join([
            f"- {self._get_entity_name(r.get('source'))} {r.get('type')} {self._get_entity_name(r.get('target'))}: {r.get('description', 'No description')}"
            for r in relevant_graph.get("relationships", [])
        ])
        
        # If no knowledge graph info, use regular context
        if not entities_text:
            entities_text = "No relevant entities found."
        if not relationships_text:
            relationships_text = "No relevant relationships found."
        
        # Generate enhanced answer
        prompt = ENHANCED_ANSWER_PROMPT.format(
            context=context,
            entities=entities_text,
            relationships=relationships_text,
            question=question
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.message.content
        except Exception as e:
            print(f"Error generating enhanced answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _get_entity_name(self, entity_id: str) -> str:
        """Get entity name from id."""
        if self.use_neo4j:
            try:
                result = self.graph.query(
                    "MATCH (e {id: $id}) RETURN e.name as name",
                    {"id": entity_id}
                )
                if result and result[0].get("name"):
                    return result[0]["name"]
                return entity_id
            except:
                return entity_id
        else:
            entity = self.in_memory_graph["entities"].get(entity_id, {})
            return entity.get("name", entity_id) 
from flask import Flask, request, jsonify, render_template
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
import openai
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
import base64
import io
import time
import numpy as np
import json
import re

app = Flask(__name__)
load_dotenv()

# Ensure log directory exists
log_dir = "/app/logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "performance_report.log")

# Performance logger setup
performance_logger = logging.getLogger("performance")
performance_logger.setLevel(logging.INFO)
if not performance_logger.hasHandlers():
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    performance_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    performance_logger.addHandler(console_handler)

# Azure credentials from Key Vault or .env
try:
    credential = DefaultAzureCredential()
    keyvault_client = SecretClient(vault_url=os.getenv("KEYVAULT_URL"), credential=credential)
    doc_endpoint = keyvault_client.get_secret("DOC-INTELLIGENCE-ENDPOINT").value
    doc_key = keyvault_client.get_secret("DOC-INTELLIGENCE-KEY").value
    openai_endpoint = keyvault_client.get_secret("OPENAI-ENDPOINT").value
    openai_key = keyvault_client.get_secret("OPENAI-KEY").value
    blob_connection_string = keyvault_client.get_secret("BLOB-CONNECTION-STRING").value
    cosmos_connection_string = keyvault_client.get_secret("COSMOS-CONNECTION-STRING").value
    search_endpoint = keyvault_client.get_secret("SEARCH-ENDPOINT").value
    search_key = keyvault_client.get_secret("SEARCH-KEY").value
except Exception as e:
    performance_logger.warning(f"Failed to retrieve secrets from Key Vault, falling back to .env: {str(e)}")
    doc_endpoint = os.getenv("DOC-INTELLIGENCE-ENDPOINT", "mock_endpoint")
    doc_key = os.getenv("DOC-INTELLIGENCE-KEY", "mock_key")
    openai_endpoint = os.getenv("OPENAI-ENDPOINT", "mock_openai_endpoint")
    openai_key = os.getenv("OPENAI-KEY", "mock_openai_key")
    blob_connection_string = os.getenv("BLOB-CONNECTION-STRING", "mock_blob_connection")
    cosmos_connection_string = os.getenv("COSMOS-CONNECTION-STRING", "mock_cosmos_connection")
    search_endpoint = os.getenv("SEARCH-ENDPOINT", "mock_search_endpoint")
    search_key = os.getenv("SEARCH-KEY", "mock_search_key")

# Azure SDK Clients
try:
    doc_client = DocumentAnalysisClient(doc_endpoint, AzureKeyCredential(doc_key))
    try:
        blob_service = BlobServiceClient.from_connection_string(blob_connection_string)
        performance_logger.info("BlobServiceClient initialized successfully")
    except Exception as e:
        performance_logger.error(f"Failed to initialize BlobServiceClient: {str(e)}. Connection string: {blob_connection_string[:20]}...")
        blob_service = None
    cosmos_client = CosmosClient.from_connection_string(cosmos_connection_string)
    database = cosmos_client.get_database_client("InvoicesDB")
    container = database.get_container_client("ExtractedData")
    search_client = SearchClient(search_endpoint, "invoices-index", AzureKeyCredential(search_key))
    index_client = SearchIndexClient(search_endpoint, AzureKeyCredential(search_key))
except Exception as e:
    performance_logger.warning(f"Using mock clients for local testing: {str(e)}")
    doc_client = None
    blob_service = None
    database = None
    container = None
    search_client = None
    index_client = None

# OpenAI setup
openai.api_type = "azure"
openai.api_base = openai_endpoint
openai.api_version = "2023-05-15"
openai.api_key = openai_key

# Create Azure AI Search index with filterable fields
def create_search_index():
    if index_client is None:
        performance_logger.warning("Skipping index creation due to mock client")
        return
    try:
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-profile"
            ),
            SimpleField(name="amountdue", type=SearchFieldDataType.Double, filterable=True),
            SimpleField(name="subtotal", type=SearchFieldDataType.Double, filterable=True),
            SearchableField(name="vendorname", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="invoicedate", type=SearchFieldDataType.DateTimeOffset, filterable=True)
        ]
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="my-hnsw"
            )],
            algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")]
        )
        index = SearchIndex(name="invoices-index", fields=fields, vector_search=vector_search)
        index_client.create_or_update_index(index)
        performance_logger.info("Search index created or updated")
    except Exception as e:
        performance_logger.error(f"Index creation failed: {str(e)}")

create_search_index()

ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_currency_value(field):
    return f"{field.currency_code}{field.amount:.2f}" if hasattr(field, 'amount') and hasattr(field, 'currency_code') else str(field)

def format_field(field, field_name=""):
    if not field:
        return {"value": "", "confidence": 0.0}
    confidence = getattr(field, 'confidence', 0.0)
    value = ""
    if hasattr(field, 'value'):
        val = field.value
        if isinstance(val, str): value = val
        elif isinstance(val, datetime): value = str(val.date())
        elif isinstance(val, dict) and 'street_address' in val:
            value = f"{val.get('street_address', '')}, {val.get('city', '')}, {val.get('state', '')} {val.get('postal_code', '')}, {val.get('country', '')}".strip(", ")
        elif isinstance(val, dict) and 'amount' in val:
            value = format_currency_value(field)
        elif isinstance(val, (int, float)):
            value = str(val)
        else: value = str(val)
    return {"value": value, "confidence": confidence}

def parse_amount(amount_str):
    """Extract numeric amount from string like 'USD610.00'"""
    if not amount_str or not isinstance(amount_str, str):
        return 0.0
    match = re.match(r'[A-Z]*(\d+\.\d{2})', amount_str)
    return float(match.group(1)) if match else 0.0

def parse_date(date_str):
    """Parse date string to ISO format"""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').isoformat()
    except ValueError:
        return None

def get_embedding(text):
    if not text or not isinstance(text, str):
        performance_logger.error("Embedding Error: Input text is empty or invalid")
        return None
    try:
        response = openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=text
        )
        embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        performance_logger.info(f"Successfully generated embedding for text: {text[:50]}...")
        return embedding
    except openai.error.AuthenticationError as e:
        performance_logger.error(f"Embedding Authentication Error: Invalid API key or endpoint. {str(e)}")
        return None
    except openai.error.InvalidRequestError as e:
        performance_logger.error(f"Embedding Invalid Request Error: Model 'text-embedding-ada-002' not found or misconfigured. {str(e)}")
        return None
    except openai.error.RateLimitError as e:
        performance_logger.error(f"Embedding Rate Limit Error: Quota exceeded for Azure OpenAI resource. {str(e)}")
        return None
    except Exception as e:
        performance_logger.error(f"Embedding General Error: {str(e)}. Endpoint: {openai.api_base}, Key: {openai.api_key[:5]}...")
        return None

def add_to_search_index(doc_id, text, embedding, metadata):
    if search_client is None:
        performance_logger.warning("Skipping search index update due to mock client")
        return
    try:
        document = {
            "id": doc_id,
            "content": text,
            "embedding": embedding.tolist(),
            "amountdue": metadata.get("amountdue"),
            "subtotal": metadata.get("subtotal"),
            "vendorname": metadata.get("vendorname"),
            "invoicedate": metadata.get("invoicedate")
        }
        search_client.upload_documents([document])
        performance_logger.info(f"Document {doc_id} added to search index")
    except Exception as e:
        performance_logger.error(f"Search index update failed: {str(e)}")

def parse_query(query):
    """Parse natural language query to Azure AI Search filter"""
    query = query.lower()
    filter_str = ""
    
    # Map field names to search index fields
    field_map = {
        "amount due": "amountdue",
        "subtotal": "subtotal",
        "vendor": "vendorname",
        "invoice date": "invoicedate"
    }
    
    # Supported operators
    operators = {
        "less than": "lt",
        "greater than": "gt",
        "equals": "eq",
        "=": "eq",
        "<": "lt",
        ">": "gt"
    }
    
    # Try to extract field, operator, and value
    for field_alias, field_name in field_map.items():
        for op_name, op in operators.items():
            pattern = rf"{field_alias}\s*(?:{op_name}|{op})\s*['\"]?([^'\"]+)['\"]?"
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field_name in ["amountdue", "subtotal"]:
                    value = parse_amount(value.replace("$", ""))
                    filter_str = f"{field_name} {op} {value}"
                elif field_name == "invoicedate":
                    value = parse_date(value)
                    if value:
                        filter_str = f"{field_name} {op} {value}"
                else:  # vendorname
                    filter_str = f"{field_name} eq '{value}'"
                break
        if filter_str:
            break
    
    # If no filter, use query as search text
    search_text = query if not filter_str else "*"
    return search_text, filter_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    start_time = time.time()
    try:
        if 'file' not in request.files:
            raise Exception("No file uploaded")
        file = request.files['file']
        doc_id = request.form['doc_id']
        if not file or not allowed_file(file.filename):
            raise Exception("Unsupported file format.")

        file_extension = file.filename.rsplit('.', 1)[1].lower()
        blob_name = f"{doc_id}.{file_extension}"

        # Save file
        if blob_service is None:
            upload_dir = "/home/uploads" if (os.getenv("WEBSITE_HOSTNAME") or os.getenv("CONTAINER_NAME")) else "/app/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, blob_name)
            file.seek(0)
            with open(file_path, 'wb') as f:
                f.write(file.read())
            file.seek(0)
        else:
            blob_client = blob_service.get_blob_client(container="invoices", blob=blob_name)
            file.seek(0)
            blob_client.upload_blob(file, overwrite=True)
            file.seek(0)

        # Mock document analysis
        if doc_client is None:
            extracted_data = {
                "id": doc_id,
                "file_extension": file_extension,
                "items": [],
                "vendorname": {"value": "Mock Vendor", "confidence": 0.9},
                "invoicetotal": {"value": "USD100.00", "confidence": 0.95},
                "average_confidence": 0.925
            }
        else:
            extract_start = time.time()
            poller = doc_client.begin_analyze_document("prebuilt-invoice", file)
            result = poller.result()
            extract_end = time.time()

            extracted_data = {
                "id": doc_id,
                "file_extension": file_extension,
                "items": []
            }
            fields_to_extract = [
                "VendorName", "CustomerName", "CustomerId", "InvoiceId", "InvoiceDate",
                "DueDate", "PurchaseOrder", "SubTotal", "TotalTax", "InvoiceTotal",
                "AmountDue", "VendorAddress", "CustomerAddress", "BillingAddress", "ShippingAddress"
            ]

            total_confidence = 0.0
            count = 0

            if result.documents:
                for invoice in result.documents:
                    for field in fields_to_extract:
                        data = format_field(invoice.fields.get(field), field)
                        extracted_data[field.lower()] = data
                        total_confidence += data['confidence']
                        count += 1
                    items_field = invoice.fields.get("Items")
                    if items_field and isinstance(items_field.value, list):
                        for item in items_field.value:
                            item_data = {
                                "description": format_field(item.value.get("Description"), "Item.Description"),
                                "quantity": format_field(item.value.get("Quantity"), "Item.Quantity"),
                                "unit_price": format_field(item.value.get("UnitPrice"), "Item.UnitPrice"),
                                "amount": format_field(item.value.get("Amount"), "Item.Amount")
                            }
                            extracted_data["items"].append(item_data)

            extracted_data['average_confidence'] = (total_confidence / count) if count else 0.0

        # Store in Cosmos DB
        if container is None:
            cosmos_dir = "/home/cosmos_mock" if (os.getenv("WEBSITE_HOSTNAME") or os.getenv("CONTAINER_NAME")) else "/app/cosmos_mock"
            os.makedirs(cosmos_dir, exist_ok=True)
            with open(f"{cosmos_dir}/{doc_id}.json", 'w') as f:
                json.dump(extracted_data, f)
        else:
            container.upsert_item(extracted_data)

        # Generate and store embeddings
        text_to_embed = json.dumps({k: v for k, v in extracted_data.items() if not k.startswith('_')})
        embedding = get_embedding(text_to_embed)
        metadata = {
            "amountdue": parse_amount(extracted_data.get('amountdue', {}).get('value', '')),
            "subtotal": parse_amount(extracted_data.get('subtotal', {}).get('value', '')),
            "vendorname": extracted_data.get('vendorname', {}).get('value', ''),
            "invoicedate": parse_date(extracted_data.get('invoicedate', {}).get('value', ''))
        }
        if embedding is not None:
            add_to_search_index(doc_id, text_to_embed, embedding, metadata)
        else:
            performance_logger.warning(f"Failed to generate embedding for document {doc_id}, skipping search index update")

        end_time = time.time()
        performance_logger.info(f"{doc_id} | Upload Time: {round(end_time - start_time, 2)}s | Extraction Time: {round(extract_end - extract_start, 2) if doc_client else 0}s | Confidence: {extracted_data['average_confidence']:.4f}")

        return jsonify({"message": "Document processed", "doc_id": doc_id, "confidence": extracted_data['average_confidence']})
    except Exception as e:
        performance_logger.error(f"Upload Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_document():
    try:
        query_start = time.time()
        query = request.form['query']
        doc_id = request.form['doc_id']

        if not query or not isinstance(query, str):
            raise Exception("Query is empty or invalid")

        # Retrieve document data
        if container is None:
            cosmos_dir = "/home/cosmos_mock" if (os.getenv("WEBSITE_HOSTNAME") or os.getenv("CONTAINER_NAME")) else "/app/cosmos_mock"
            with open(f"{cosmos_dir}/{doc_id}.json", 'r') as f:
                doc_data = json.load(f)
        else:
            doc_data = container.read_item(item=doc_id, partition_key=doc_id)
        filtered_data = {k: v for k, v in doc_data.items() if not k.startswith('_')}

        # Get file for image-based context
        file_extension = doc_data.get('file_extension', 'pdf')
        blob_name = f"{doc_id}.{file_extension}"
        if blob_service is None:
            upload_dir = "/home/uploads" if (os.getenv("WEBSITE_HOSTNAME") or os.getenv("CONTAINER_NAME")) else "/app/uploads"
            file_path = os.path.join(upload_dir, blob_name)
            with open(file_path, 'rb') as f:
                blob_data = f.read()
        else:
            blob_client = blob_service.get_blob_client(container="invoices", blob=blob_name)
            blob_data = blob_client.download_blob().readall()
        image_base64 = base64.b64encode(blob_data).decode('utf-8')
        mime_type = 'application/pdf' if file_extension == 'pdf' else 'image/jpeg'

        # RAG: Retrieve relevant documents
        query_embedding = get_embedding(query)
        if query_embedding is None:
            raise Exception("Failed to generate query embedding")

        results = search_client.search(
            search_text=None,
            vector_queries=[{"vector": query_embedding.tolist(), "kind": "vector", "fields": "embedding", "k": 3}]
        )
        retrieved_context = [json.loads(result['content']) for result in results]

        context = f"Current document: {json.dumps(filtered_data)}\nRelevant documents: {json.dumps(retrieved_context)}"

        response = openai.ChatCompletion.create(
            engine="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that answers questions based on provided invoice data. Use the context to provide accurate answers."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Context: {context}\nQuestion: {query}"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        query_end = time.time()

        performance_logger.info(f"{doc_id} | Query Time: {round(query_end - query_start, 2)}s | Query: {query}")
        return jsonify({"answer": answer, "source": filtered_data, "retrieved_context": retrieved_context})
    except Exception as e:
        performance_logger.error(f"Query Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query_all', methods=['POST'])
def query_all():
    try:
        query_start = time.time()
        query = request.form['query']

        if not query or not isinstance(query, str):
            raise Exception("Query is empty or invalid")

        if search_client is None:
            performance_logger.warning("Returning empty results due to mock client")
            return jsonify({"invoices": []})

        # Parse query to get search text and filter
        search_text, filter_str = parse_query(query)
        
        # Search with filter and/or text
        results = search_client.search(
            search_text=search_text,
            filter=filter_str if filter_str else None,
            select=["id", "content", "amountdue", "subtotal", "vendorname", "invoicedate"],
            top=50
        )
        invoices = []
        for result in results:
            content = json.loads(result['content'])
            invoices.append({
                "doc_id": result['id'],
                "vendorname": content.get('vendorname', {}).get('value', ''),
                "amountdue": content.get('amountdue', {}).get('value', ''),
                "subtotal": content.get('subtotal', {}).get('value', ''),
                "invoicedate": content.get('invoicedate', {}).get('value', '')
            })

        query_end = time.time()
        performance_logger.info(f"Query All Time: {round(query_end - query_start, 2)}s | Query: {query} | Results: {len(invoices)}")

        return jsonify({"invoices": invoices})
    except Exception as e:
        performance_logger.error(f"Query All Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

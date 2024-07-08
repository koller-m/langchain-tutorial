const axios = require("axios");
const { UnstructuredLoader } = require("langchain/document_loaders/fs/unstructured");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { Document } = require("langchain/document");
const { RetrievalQAChain } = require("langchain/chains");
const { ChatOpenAI } = require("@langchain/openai");
const dotenv = require("dotenv");
const fs = require("fs");
const path = require("path");

dotenv.config()

const config = {
    domain: "https://news.ycombinator.com",
    query: "What is the second story on hacker news?"
}

let currentStep = 1
const startTime = performance.now()

function logTimeWriteOutStep(message, response=null) {
    const elapsedTime = `[${((performance.now() - startTime) / 1000).toFixed(2)}s]`
    const logMessage = `${elapsedTime} Step ${currentStep}: ${message}`
    console.log(logMessage);
    response ? console.log(response) : null
    currentStep += 1
}

async function main() {
    logTimeWriteOutStep("Starting main function")

    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY,
        batchSize: 512,
    })

    logTimeWriteOutStep("OpenAI Embeddings initialized")

    const response = await axios.get(config.domain)
    logTimeWriteOutStep("GET request to Hacker News completed")

    const cacheDir = path.join(__dirname, "cache")
    if (!fs.existsSync(cacheDir)) {
        fs.mkdirSync(cacheDir)
    }

    logTimeWriteOutStep("Cache directory checked/created")

    const filePath = path.join(cacheDir, "response.html")
    fs.writeFileSync(filePath, response.data)

    logTimeWriteOutStep("Response data saved as HTML file")

    const loader = new UnstructuredLoader(filePath, { 
        apiKey: process.env.UNSTRUCTURED_API_KEY, 
        apiUrl: process.env.UNSTRUCTURED_URL })
    const loadedData = await loader.load()

    logTimeWriteOutStep("HTML file loaded")

    const docs = loadedData.map((item) => new Document({ pageContent: item.pageContent, metadata: item.metadata }))

    logTimeWriteOutStep("Loaded data converted into Document objects")

    const vectorStore = await HNSWLib.fromDocuments(docs, embeddings)

    logTimeWriteOutStep("Vector store created with embeddings")

    const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" })
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    returnSourceDocuments: true,
    })
    logTimeWriteOutStep("QA Chain set up")

    const resp = await chain.call({ query: config.query })
    logTimeWriteOutStep("QA Response", resp.text)
    logTimeWriteOutStep("Execution timing ended")
}

main()
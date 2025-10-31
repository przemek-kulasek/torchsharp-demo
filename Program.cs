// Import TorchSharp - C# bindings for PyTorch deep learning library

using static TorchSharp.torch; // Static imports allow us to use tensor operations directly
using TorchSharp; // Main TorchSharp namespace
using TorchSharp.Modules; // Neural network modules (layers, activations, etc.)

// =============================================================================
// HYPERPARAMETERS - These control the model architecture and training behavior
// =============================================================================

const int blockSize = 64; // Maximum sequence length the model can process at once
// This is the "context window" - how far back the model looks
const int batchSize = 16; // Number of training examples processed simultaneously  
// Larger = more stable gradients but needs more GPU memory
const int maxIters = 2000; // Total number of training steps
const int evalInterval = 200; // How often to evaluate model performance
const double learningRate = 1e-3; // Step size for gradient descent (0.001)
// Higher = faster learning but less stable
const int nEmbedding = 128; // Dimension of embedding vectors (how we represent each token)
// Higher = more expressive but needs more computation
const int nHead = 4; // Number of attention heads in multi-head attention
// Each head learns different types of relationships
const int nLayer = 10; // Number of transformer blocks stacked together
// More layers = deeper model that can learn more complex patterns
const int ffHiddenMult = 4; // Feed-forward network hidden layer size multiplier
// Hidden size = nEmbedding * ffHiddenMult
const int maxNewTokens = 200; // Maximum tokens to generate during text generation

Console.WriteLine("Mini GPT Training with TorchSharp - Algorithm Analysis");
Console.WriteLine("=".PadRight(60, '='));

// =============================================================================
// DEVICE SELECTION - Choose between CPU and GPU for computation
// =============================================================================
var device = cuda.is_available() ? CUDA : CPU; // Use GPU if available, otherwise CPU
Console.WriteLine($"Device: {device.type}"); // GPU is ~100x faster for deep learning

// =============================================================================
// DATA LOADING AND PREPROCESSING
// =============================================================================

// Read the entire text file - this is our training dataset
string text = File.ReadAllText("input.txt");
Console.WriteLine($"Loaded input.txt - {text.Length} characters");

// CHARACTER-LEVEL TOKENIZATION
// We treat each unique character as a separate token (vocabulary item)
var chars = text.Distinct().OrderBy(c => c).ToArray(); // Get all unique characters, sorted
var vocabSize = chars.Length; // This is our vocabulary size (typically ~65 for English text)

// Create mappings between characters and token IDs
var stoi = new Dictionary<char, int>(); // String-to-integer: 'A' -> 0, 'B' -> 1, etc.
var itos = new Dictionary<int, char>(); // Integer-to-string: 0 -> 'A', 1 -> 'B', etc.
for (int i = 0; i < chars.Length; i++)
{
    stoi[chars[i]] = i; // Map character to unique integer ID
    itos[i] = chars[i]; // Map integer ID back to character
}

Console.WriteLine($"Vocab size: {vocabSize}");

// CONVERT TEXT TO NUMERICAL DATA
// Neural networks work with numbers, not text, so we convert each character to its token ID
var data = text.Select(c => (long)stoi[c]).ToArray(); // Convert entire text to array of token IDs

// TRAIN/VALIDATION SPLIT
// Use 90% for training, 10% for validation (testing how well model generalizes)
var trainData = data.Take((int)(data.Length * 0.9)).ToArray(); // First 90% for training
var valData = data.Skip((int)(data.Length * 0.9)).ToArray(); // Last 10% for validation

// =============================================================================
// HELPER FUNCTIONS FOR DATA PROCESSING
// =============================================================================

// ENCODE TEXT TO TOKENS
// Convert human-readable text into token IDs that the model can process
long[] EncodeText(string inputText)
{
    Console.WriteLine($"📝 Encoding: '{inputText}'");
    // For each character, look up its token ID (or use space if character not in vocabulary)
    var tokens = inputText.Select(c => stoi.ContainsKey(c) ? (long)stoi[c] : (long)stoi[' ']).ToArray();
    Console.WriteLine($"   Tokens: [{string.Join(", ", tokens)}]");
    Console.WriteLine($"   Chars:  [{string.Join(", ", tokens.Select(t => $"'{itos[(int)t]}'"))}]");
    return tokens;
}

// DECODE TOKENS TO TEXT
// Convert token IDs back into human-readable text
string DecodeTokens(long[] tokens)
{
    // For each token ID, look up its character (or use '?' if invalid token)
    return string.Join("", tokens.Select(token => itos.ContainsKey((int)token) ? itos[(int)token] : '?'));
}

// BATCH DATA LOADING FOR TRAINING
// During training, we process multiple sequences simultaneously for efficiency
(Tensor, Tensor) GetBatch(long[] data)
{
    // Randomly select starting positions for batchSize sequences
    var indices = randint(0, data.Length - blockSize, new long[] { batchSize }, device: device);

    // Create input (x) and target (y) tensors
    // x contains the input sequences, y contains what the model should predict (x shifted by 1)
    var x = zeros(new long[] { batchSize, blockSize }, dtype: ScalarType.Int64, device: device);
    var y = zeros(new long[] { batchSize, blockSize }, dtype: ScalarType.Int64, device: device);

    for (int i = 0; i < batchSize; i++)
    {
        var ix = indices[i].ToInt32(); // Starting position for this sequence
        // Input sequence: tokens from ix to ix+blockSize
        x[i] = tensor(data.Skip(ix).Take(blockSize).ToArray(), dtype: ScalarType.Int64, device: device);
        // Target sequence: tokens from ix+1 to ix+blockSize+1 (shifted by 1 position)
        y[i] = tensor(data.Skip(ix + 1).Take(blockSize).ToArray(), dtype: ScalarType.Int64, device: device);
    }

    return (x, y); // Return input-target pairs for training
}

// =============================================================================
// MODEL INITIALIZATION AND TRAINING SETUP
// =============================================================================

// Create the GPT model with specified architecture
var model = new GPTModel(vocabSize, nEmbedding, nHead, nLayer, blockSize).to(device);

// AdamW optimizer - advanced version of gradient descent with momentum and weight decay
// Automatically adjusts learning rates and helps with training stability
var optimizer = optim.AdamW(model.parameters(), learningRate);

// Count total number of trainable parameters in the model
Console.WriteLine($"Model parameters: {model.parameters().Sum(p => p.numel())}");

// =============================================================================
// TRAINING LOOP - WHERE THE NEURAL NETWORK LEARNS
// =============================================================================

Console.WriteLine("\nTraining for demonstration...");
for (int iter = 0; iter < Math.Min(maxIters, 500); iter++) // Train for 500 steps (shortened for demo)
{
    // FORWARD PASS - Make predictions
    var (xb, yb) = GetBatch(trainData); // Get a random batch of training data
    var logits = model.forward(xb); // Run input through model to get predictions

    // logits shape: [batchSize, blockSize, vocabSize]
    // Each position predicts probability distribution over entire vocabulary
    var B = logits.shape[0]; // Batch size
    var T = logits.shape[1]; // Sequence length (blockSize)
    var C = logits.shape[2]; // Vocabulary size

    // LOSS CALCULATION - How wrong are our predictions?
    // Cross-entropy loss: measures difference between predicted and actual next tokens
    var loss = nn.functional.cross_entropy(
        logits.view(B * T, C), // Reshape predictions: [B*T, C] 
        yb.view(B * T) // Reshape targets: [B*T]
    );

    // BACKWARD PASS - Calculate gradients (how to improve)
    optimizer.zero_grad(); // Clear previous gradients
    loss.backward(); // Calculate gradients using backpropagation
    optimizer.step(); // Update model parameters using calculated gradients

    if (iter % 100 == 0)
        Console.WriteLine($"Step {iter}: loss {loss.ToDouble():F4}"); // Print progress
}

Console.WriteLine("\nTraining completed! Now analyzing how prompts affect generation...\n");

// =============================================================================
// ALGORITHM EXPLANATION FOR EDUCATIONAL PURPOSES
// =============================================================================

Console.WriteLine("🧠 HOW THE ALGORITHM WORKS:");
Console.WriteLine(@"
1. TOKENIZATION: Your prompt ""HAMLET:"" becomes [25, 8, 40, 36, 22, 47, 10]
2. EMBEDDINGS: Each token → 128-dim vector + position info
3. ATTENTION: Model looks at relationships between characters
4. PREDICTION: Outputs probabilities for all 65 possible next characters  
5. SAMPLING: Randomly selects based on probabilities
6. REPEAT: Adds selected character and continues
");

// =============================================================================
// DETAILED GENERATION ANALYSIS FUNCTION
// =============================================================================

// This function shows step-by-step how the model generates text
Tensor AnalyzeGeneration(string prompt, int steps = 5)
{
    Console.WriteLine($"\n🔍 ANALYZING: '{prompt}'");
    Console.WriteLine("-".PadRight(40, '-'));

    Tensor context;
    if (string.IsNullOrEmpty(prompt))
    {
        // Start with a single zero token (empty context)
        context = zeros(new long[] { 1, 1 }, dtype: ScalarType.Int64, device: device);
        Console.WriteLine("Starting from empty context (cold start)");
    }
    else
    {
        // Convert prompt to tokens and create tensor
        var tokens = EncodeText(prompt);
        context = tensor(tokens, dtype: ScalarType.Int64, device: device).unsqueeze(0); // Add batch dimension
    }

    model.eval(); // Set model to evaluation mode (disables dropout, etc.)

    // Generate tokens step by step
    for (int i = 0; i < steps; i++)
    {
        // Crop context to maximum sequence length model can handle
        var contextCrop = context.shape[1] <= blockSize ? context : context[.., ^blockSize..];
        var currentTokens = contextCrop[0].to(CPU).data<long>().ToArray();
        var currentText = DecodeTokens(currentTokens);

        Console.WriteLine($"\nStep {i + 1}:");
        Console.WriteLine($"  📖 Context: '{currentText}'");

        // FORWARD PASS - Get model's predictions
        var logits = model.forward(contextCrop); // Run context through model
        var logitsLast = logits[0, -1, ..]; // Get predictions for next token (last position)

        // ANALYZE TOP PREDICTIONS
        // Show the 5 most likely next characters and their probabilities
        var topK = torch.topk(logitsLast, 5); // Get top 5 predictions
        var topTokens = topK.indices.to(CPU).data<long>().ToArray(); // Token IDs
        var topProbs = torch.softmax(topK.values, dim: 0).to(CPU).data<float>().ToArray(); // Probabilities 
        Console.WriteLine("  🎯 Top predictions:");
        for (int j = 0; j < 5; j++)
        {
            var token = (int)topTokens[j];
            var prob = topProbs[j] * 100; // Convert to percentage
            var character = itos[token];
            var displayChar = character == '\n' ? "\\n" : character.ToString(); // Handle newlines
            Console.WriteLine($"   '{displayChar}' → {prob:F1}%");
        }

        // SAMPLING - Select next token
        // Convert logits to probabilities and randomly sample based on those probabilities
        var probs = softmax(logitsLast, dim: 0); // Convert to probability distribution
        var nextToken = multinomial(probs, 1); // Sample from distribution
        var selectedToken = (int)nextToken.to(CPU).data<long>()[0];
        var selectedChar = itos[selectedToken];
        var selectedDisplay = selectedChar == '\n' ? "\\n" : selectedChar.ToString();

        Console.WriteLine($"  ✅ Selected: '{selectedDisplay}' (token {selectedToken})");

        // ADD TO CONTEXT - Append selected token and continue
        context = cat(new[] { context, nextToken.unsqueeze(0) }, dim: 1);
    }

    return context;
}

// =============================================================================
// DEMONSTRATION OF DIFFERENT PROMPT TYPES
// =============================================================================

// Test different types of prompts to show how context affects generation
var examples = new (string prompt, string explanation)[]
{
    ("HAMLET:", "Character name - triggers dialogue patterns"),
    ("To be or not", "Famous quote - model completes known phrases"),
    ("The king", "Common phrase - continues with royal/formal language"),
    ("", "Empty prompt - generates from general learned patterns")
};

foreach (var (prompt, explanation) in examples)
{
    Console.WriteLine($"\n{'=' * 60}");
    Console.WriteLine($"PROMPT ANALYSIS: {explanation}");
    Console.WriteLine($"{'=' * 60}");

    var result = AnalyzeGeneration(prompt, 3); // Analyze 3 generation steps
    var fullText = DecodeTokens(result[0].to(CPU).data<long>().ToArray());
    Console.WriteLine($"\n📄 FULL RESULT: '{fullText}'");
}

// =============================================================================
// INTERACTIVE MODE FOR EXPERIMENTATION
// =============================================================================

Console.WriteLine($"\n{'=' * 60}");
Console.WriteLine("INTERACTIVE ANALYSIS MODE");
Console.WriteLine($"{'=' * 60}");
Console.WriteLine("Enter prompts to see how they influence generation:");

while (true)
{
    Console.Write("\n🎭 Your prompt (or 'quit'): ");
    var userPrompt = Console.ReadLine();

    if (string.IsNullOrEmpty(userPrompt) || userPrompt.ToLower() == "quit")
        break;

    try
    {
        // Analyze the prompt step by step
        var result = AnalyzeGeneration(userPrompt, 5);
        // Generate more text for full context
        var fullGenerated = model.Generate(result, 50);
        var fullText = DecodeTokens(fullGenerated[0].to(CPU).data<long>().ToArray());
        Console.WriteLine($"\n📖 Extended result: {fullText}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Error: {ex.Message}");
    }
}

Console.WriteLine("\n🎉 Analysis complete! You can see how your prompts influence the model's predictions.");

// =============================================================================
// NEURAL NETWORK MODEL CLASSES
// =============================================================================

// MULTI-HEAD ATTENTION - The core of the transformer architecture
// This mechanism allows the model to focus on different parts of the input simultaneously
public class MultiHeadAttention : nn.Module<Tensor, Tensor>
{
    // Linear projection layers for keys, queries, values, and output
    private readonly Linear keyProj, queryProj, valueProj, outProj;
    private readonly Dropout dropout; // Regularization to prevent overfitting
    private readonly int nHead, headSize; // Number of attention heads and size of each head

    public MultiHeadAttention(int nEmbed, int nHead, double dropout = 0.1) : base("MultiHeadAttention")
    {
        this.nHead = nHead;
        this.headSize = nEmbed / nHead; // Each head gets a portion of the embedding dimension

        // Linear layers to project embeddings into key, query, and value spaces
        keyProj = nn.Linear(nEmbed, nEmbed, hasBias: false); // Keys: what information is available
        queryProj = nn.Linear(nEmbed, nEmbed, hasBias: false); // Queries: what information we're looking for  
        valueProj = nn.Linear(nEmbed, nEmbed, hasBias: false); // Values: the actual information content
        outProj = nn.Linear(nEmbed, nEmbed); // Final output projection
        this.dropout = nn.Dropout(dropout);
        RegisterComponents(); // Register all layers so they get trained
    }

    public override Tensor forward(Tensor x)
    {
        var (B, T, C) = (x.shape[0], x.shape[1], x.shape[2]); // Batch, Time, Channels

        // PROJECT TO KEY, QUERY, VALUE SPACES
        // Each position creates keys (what it offers), queries (what it wants), values (what it contains)
        var k = keyProj.forward(x).view(B, T, nHead, headSize).transpose(1, 2); // [B, nH, T, hS]
        var q = queryProj.forward(x).view(B, T, nHead, headSize).transpose(1, 2); // [B, nH, T, hS]  
        var v = valueProj.forward(x).view(B, T, nHead, headSize).transpose(1, 2); // [B, nH, T, hS]

        // ATTENTION MECHANISM - Calculate which positions should attend to which
        // Attention = softmax(Q * K^T / sqrt(d_k)) * V
        var wei = q.matmul(k.transpose(-2, -1)) * (1.0 / Math.Sqrt(headSize)); // Attention weights [B, nH, T, T]

        // CAUSAL MASKING - Prevent looking at future tokens (for autoregressive generation)
        // Lower triangular mask ensures each position can only see previous positions
        var mask = ones(T, T, device: x.device).tril(); // Lower triangular matrix of 1s
        wei = wei.masked_fill(mask == 0, float.NegativeInfinity); // Set upper triangle to -inf
        wei = softmax(wei, dim: -1); // Convert to probabilities (each row sums to 1)
        wei = dropout.forward(wei); // Apply dropout for regularization

        // APPLY ATTENTION TO VALUES
        // Weighted combination of values based on attention weights  
        var output = wei.matmul(v); // [B, nH, T, hS] - Attend to values
        output = output.transpose(1, 2).contiguous().view(B, T, C); // Reshape back to [B, T, C]
        return outProj.forward(output); // Final linear projection
    }
}

// FEED-FORWARD NETWORK - Processes each position independently  
// This adds non-linearity and helps the model learn complex patterns
public class FeedForward : nn.Module<Tensor, Tensor>
{
    private readonly Linear linear1, linear2; // Two linear layers with ReLU activation between
    private readonly Dropout dropout; // Regularization

    public FeedForward(int nEmbed, int ffHiddenMult = 4, double dropout = 0.1) : base("FeedForward")
    {
        var hiddenSize = nEmbed * ffHiddenMult; // Hidden layer is typically 4x embedding size
        linear1 = nn.Linear(nEmbed, hiddenSize); // Expand to larger dimension
        linear2 = nn.Linear(hiddenSize, nEmbed); // Project back to original dimension
        this.dropout = nn.Dropout(dropout);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // STANDARD FEEDFORWARD: Linear -> ReLU -> Dropout -> Linear
        x = linear1.forward(x); // Expand dimension and transform
        x = nn.functional.relu(x); // Apply ReLU activation (introduces non-linearity)
        x = dropout.forward(x); // Apply dropout for regularization
        x = linear2.forward(x); // Project back to original dimension
        return x;
    }
}

// TRANSFORMER BLOCK - Combines attention and feed-forward with residual connections
// This is the fundamental building block of transformer models like GPT
public class TransformerBlock : nn.Module<Tensor, Tensor>
{
    private readonly MultiHeadAttention attention; // Self-attention mechanism
    private readonly FeedForward feedForward; // Feed-forward network
    private readonly LayerNorm ln1, ln2; // Layer normalization for training stability

    public TransformerBlock(int nEmbed, int nHead, int ffHiddenMult) : base("TransformerBlock")
    {
        attention = new MultiHeadAttention(nEmbed, nHead);
        feedForward = new FeedForward(nEmbed, ffHiddenMult);
        ln1 = nn.LayerNorm(nEmbed); // Normalize before attention
        ln2 = nn.LayerNorm(nEmbed); // Normalize before feed-forward
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // PRE-NORM TRANSFORMER BLOCK ARCHITECTURE
        // Residual connections help with gradient flow during training
        x = x + attention.forward(ln1.forward(x)); // Self-attention with residual connection
        x = x + feedForward.forward(ln2.forward(x)); // Feed-forward with residual connection
        return x;
    }
}

// GPT MODEL - The complete language model architecture
// Combines embeddings, transformer blocks, and output projection
public class GPTModel : nn.Module<Tensor, Tensor>
{
    private readonly Embedding tokenEmbedding, positionEmbedding; // Convert tokens to vectors
    private readonly ModuleList<nn.Module<Tensor, Tensor>> blocks; // Stack of transformer blocks
    private readonly LayerNorm layerNorm; // Final layer normalization
    private readonly Linear lmHead; // Language modeling head (output projection)
    private readonly int blockSize; // Maximum sequence length

    public GPTModel(int vocabSize, int nEmbed, int nHead, int nLayer, int blockSize) : base("GPTModel")
    {
        this.blockSize = blockSize;

        // EMBEDDING LAYERS
        // Convert discrete tokens into continuous vector representations
        tokenEmbedding = nn.Embedding(vocabSize, nEmbed); // Token embeddings: token_id -> vector
        positionEmbedding = nn.Embedding(blockSize, nEmbed); // Position embeddings: position -> vector

        // TRANSFORMER BLOCKS
        // Stack multiple transformer blocks for increased model capacity
        blocks = nn.ModuleList<nn.Module<Tensor, Tensor>>();
        for (int i = 0; i < nLayer; i++)
            blocks.Add(new TransformerBlock(nEmbed, nHead, 4)); // Each block has same architecture

        // OUTPUT LAYERS
        layerNorm = nn.LayerNorm(nEmbed); // Final normalization
        lmHead = nn.Linear(nEmbed, vocabSize); // Project to vocabulary size for token prediction
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var (B, T) = (x.shape[0], x.shape[1]); // Batch size, sequence length

        // EMBEDDING LOOKUP
        // Convert token IDs to rich vector representations
        var tokEmb = tokenEmbedding.forward(x); // Token embeddings [B, T, C]
        var pos = arange(T, device: x.device); // Position indices [T]
        var posEmb = positionEmbedding.forward(pos); // Position embeddings [T, C]
        var h = tokEmb + posEmb; // Combine token + position info [B, T, C]

        // TRANSFORMER PROCESSING  
        // Pass through all transformer blocks sequentially
        foreach (var block in blocks)
            h = block.forward(h); // Each block refines the representations

        // OUTPUT PROJECTION
// Convert final hidden states to vocabulary predictions
        h = layerNorm.forward(h); // Final normalization
        return lmHead.forward(h); // Project to vocab size [B, T, vocab_size]
    }

    // TEXT GENERATION METHOD
    // Autoregressively generate new tokens by sampling from predicted distributions
    public Tensor Generate(Tensor context, int maxNewTokens)
    {
        for (int i = 0; i < maxNewTokens; i++) // Generate one token at a time
        {
            // CONTEXT WINDOWING
            // Keep only the most recent tokens if sequence gets too long
            var contextCrop = context.shape[1] <= blockSize ? context : context[.., ^blockSize..];

            // FORWARD PASS
            var logits = forward(contextCrop); // Get predictions for all positions
            var logitsLast = logits[0, -1, ..]; // Get predictions for next token (last position)

            // SAMPLING
            var probs = softmax(logitsLast, dim: 0); // Convert logits to probabilities
            var nextToken = multinomial(probs, 1); // Sample token based on probabilities

            // UPDATE CONTEXT
            // Add the newly generated token to the context for next iteration
            context = cat(new[] { context, nextToken.unsqueeze(0) }, dim: 1);
        }

        return context; // Return the extended sequence
    }
}
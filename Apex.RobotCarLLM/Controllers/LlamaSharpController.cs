using LLama.Common;
using LLama.Native;
using LLama;
using LLamaSharp.SemanticKernel.ChatCompletion;
using LLamaSharp.SemanticKernel.TextCompletion;
using Microsoft.AspNetCore.Mvc;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Planning.Handlebars;
using Microsoft.SemanticKernel.TextGeneration;
using Microsoft.SemanticKernel.Planning;
using Microsoft.SemanticKernel;
using Serilog;
using Swashbuckle.AspNetCore.Annotations;
using Apex.RobotCarLLM.Models;
using Apex.RobotCarLLM.Helpers;

namespace Apex.RobotCarLLM.Controllers;

[ApiController]
[Route("[controller]")]
public class LlamaSharpController : ControllerBase
{
    private const string RoverImagePath = @"c:\Temp\RoverImages\m12.jpeg";

    public readonly Dictionary<string, LLM> Models = new()
    {
        ["llava-mistral.Q8_0"] = new LLM(ModelTypes.LlavaLlama, @"c:\temp\LLMs\llava-v1.6-mistral-7b.Q8_0.gguf", @"c:\temp\LLMs\mmproj-mistral7b-f16.gguf"),
        ["llava-mistral.Q5_K_M"] = new LLM(ModelTypes.LlavaLlama, @"c:\temp\LLMs\llava-v1.6-mistral-7b.Q5_K_M.gguf", @"c:\temp\LLMs\mmproj-mistral7b-f16-q6_k.gguf"),
        ["llava-phi-3-f16"] = new LLM(ModelTypes.LlavaLlama, @"c:\temp\LLMs\phi\llava-phi-3-mini-f16.gguf", @"c:\temp\LLMs\phi\llava-phi-3-mini-mmproj-f16.gguf"),
        ["phi-3-instruct-function-calling"] = new LLM(ModelTypes.LlamaFunctionCalling, @"Phi-3-mini-4k-instruct-function-calling_Q8_0.gguf", @""),
        ["llava-llama-3-f16"] = new LLM(ModelTypes.LlavaLlama, @"c:\temp\LLMs\llava-llama-3\llava-llama-3-8b-v1_1-f16.gguf", @"c:\temp\LLMs\llava-llama-3\llava-llama-3-8b-v1_1-mmproj-f16.gguf"),
        ["llava-llama-3.Q4_K_M"] = new LLM(ModelTypes.LlavaLlama, @"c:\temp\LLMs\llava\llava-llama-3-8b-v1_1.Q4_K_M.gguf", @"c:\temp\LLMs\llava\mmproj-model-f16(2).gguf"),
        ["llama-3-instruct-function-calling"] = new LLM(ModelTypes.LlamaFunctionCalling, @"c:\temp\LLMs\phi\llama-3-8b-instruct-function-calling-v0.2.Q6_K.gguf.gguf", ""),
    };

    //https://github.com/microsoft/Phi-3CookBook

    /// <summary>
    /// 
    /// </summary>
    /// <param name="imagePath"></param>
    /// <param name="videoModelAlias"></param>
    /// <returns></returns>
    [HttpGet("/llamasharp/chat/image")]
    [SwaggerOperation(Summary = "Gets the details of the specified model by key.")]
    public async Task<IActionResult> GetChatCompletionWithImageAsync(string? imagePath, string videoModelAlias)
    {
        var LLM = Models?[videoModelAlias];

        Console.WriteLine("\n===========================================================================");
        Console.WriteLine($"{Path.GetFileNameWithoutExtension(LLM.ModelPath)}");
        Console.WriteLine("---------------------------------------------------------------------------");

        imagePath ??= RoverImagePath;

        NativeLibraryConfig.Instance.WithCuda(true);

        NativeLogConfig.llama_log_set((level, message) =>
        {
            if (message.StartsWith("llama_") ||
                message.StartsWith("llm_") ||
                message.StartsWith("clip_") ||
                message.StartsWith('.'))
                return;
            Console.WriteLine(message.TrimEnd('\n'));
        });

        var parameters = new ModelParams(LLM.ModelPath)
        {
            ContextSize = 4096,
            Seed = 111,
            GpuLayerCount = 8,
        };

        using var model = await LLamaWeights.LoadFromFileAsync(parameters);
        using var context = model.CreateContext(parameters);
        using var clipModel = await LLavaWeights.LoadFromFileAsync(LLM.MMProjPath);

        var executor = new InteractiveExecutor(context, clipModel);

        var imageBytes = await System.IO.File.ReadAllBytesAsync(imagePath);
        executor.Images.Add(imageBytes);
        // Each prompt with images we clear cache
        // When the prompt contains images we clear KV_CACHE to restart conversation
        // See: https://github.com/ggerganov/llama.cpp/discussions/3620
        executor.Context.NativeHandle.KvCacheRemove(LLamaSeqId.Zero, -1, -1);

        var builder = Kernel.CreateBuilder();
        builder.Services.AddKeyedSingleton<ITextGenerationService>("local-llava", new LLamaSharpTextCompletion(executor));
        var kernel = builder.Build();

        var inferenceParams = new InferenceParams
        {
            Temperature = 0.4f,
            AntiPrompts = ["USER", "ASSISTANT", "<|end|>", "user", "assistant"],
            MaxTokens = 1024
        };

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"Maximum tokens: {inferenceParams.MaxTokens} and the context size is {parameters.ContextSize}.");

        var goal = """
            <image>
            The attached image is an aerial view of you, the rover in the image.
            Your objective is to analyze the image for obstacles find a path the distant hills.
            If the objective is too complex for you, break it down into basic motor commands. The permited basic motor commands are: {{$input}}
            
            Example 1:
            user: "Avoid the boulder"
            assistant: {
              "plan": "prevent an imminent crush and resume the original direction",
              "path": [
                "turn left",
                "go forward",
                "turn right",
                "go forward"
            ]}

            Example 2:
            user: "Walk like a jellyfish"
            assistant: {
              "plan": "jellyfish uses propulsion and relaxation to produce movement",
              "path": [
                "go forward",
                "stop",
                "go forward",
                "stop"
            ]}

            Example 3:
            user: "Go on the edge of a square"
            assistant: {
              "plan": "a square has four connected edges",
              "path": [
                "turn left",
                "turn left",
                "turn left",
                "turn left"
            ]}
            
            Respond with JSON object.
            ASSISTANT:
            """;

        var executionSettings = new OpenAIPromptExecutionSettings
        {
            MaxTokens = 1024,
            Temperature = 0.1f,
            //ResponseFormat = "json_object",
            StopSequences = ["USER", "ASSISTANT", "<|end|>", "user", "assistant"]
        };
        var kernelArguments = new KernelArguments(executionSettings)
        {
            ["input"] = "forward, backward, turn left, turn right and stop."
        };
        //geologial formations
        var promptTemplateFactory = new KernelPromptTemplateFactory();
        var promptTemplateRenderer = promptTemplateFactory.Create(new PromptTemplateConfig(goal));
        var renderedPrompt = await promptTemplateRenderer.RenderAsync(kernel, kernelArguments);
        Console.WriteLine("\n-- rendered prompt -------------------------------------------------------");
        await kernel.PrintRenderedPromptAsync(renderedPrompt, kernelArguments);

        Console.WriteLine("\n-- response -------------------------------------------------");
        var result = string.Empty;

        var session = new ChatSession(executor);
        session.AddSystemMessage("""
            You are a robotic rover on a mission to explore the planet Mars.
            """);

        await foreach (var token in session.ChatAsync(new LLama.Common.ChatHistory.Message(LLama.Common.AuthorRole.User, goal), inferenceParams))
        {
            Console.Write(token);
            result += token;
        }

        Console.WriteLine("\n-- done -------------------------------------------------");

        return Ok(result);
    }

    [HttpGet("/llamasharp/handebars")]
    public async Task<IActionResult> GetHandlebarsPlannerAsync(string? imagePath, string modelAlias)
    {
        var LLM = Models?[modelAlias];

        Console.WriteLine("\n===========================================================================");
        Console.WriteLine($"{Path.GetFileNameWithoutExtension(LLM.ModelPath)}");
        Console.WriteLine("---------------------------------------------------------------------------");

        imagePath ??= RoverImagePath;

        NativeLibraryConfig.Instance.WithCuda(true);

        NativeLogConfig.llama_log_set((level, message) =>
        {
            if (message.StartsWith("llama_") ||
                message.StartsWith("llm_") ||
                message.StartsWith("clip_") ||
                message.StartsWith('.'))
                return;
            Console.WriteLine(message.TrimEnd('\n'));
        });

        var parameters = new ModelParams(LLM.ModelPath)
        {
            ContextSize = 4096,
            Seed = 111,
            GpuLayerCount = 8,
        };

        using var model = await LLamaWeights.LoadFromFileAsync(parameters);
        using var context = model.CreateContext(parameters);
        using var clipModel = await LLavaWeights.LoadFromFileAsync(LLM.MMProjPath);

        var executor = new InteractiveExecutor(context, clipModel);

        var imageBytes = await System.IO.File.ReadAllBytesAsync(imagePath);
        executor.Images.Add(imageBytes);
        // Each prompt with images we clear cache
        // When the prompt contains images we clear KV_CACHE to restart conversation
        // See: https://github.com/ggerganov/llama.cpp/discussions/3620
        executor.Context.NativeHandle.KvCacheRemove(LLamaSeqId.Zero, -1, -1);

        var builder = Kernel.CreateBuilder();
        ////builder.Services.AddKeyedSingleton<ITextGenerationService>("local-llava", new LLamaSharpTextCompletion(executor));
        builder.Services.AddKeyedSingleton<IChatCompletionService>("local-llava", new LLamaSharpChatCompletion(executor));

        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            //.MinimumLevel.Information()
            //.MinimumLevel.Warning()
            .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
            .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
            .MinimumLevel.Override("System.Net.Http", Serilog.Events.LogEventLevel.Warning)
            .WriteTo.Console()
            .CreateLogger();

        // Add services to the container.
        builder.Services.AddLogging(c => c.AddSerilog(Log.Logger));



        var kernel = builder.Build();

        ////var session = new ChatSession(executor);

        var inferenceParams = new InferenceParams
        {
            Temperature = 0.4f,
            AntiPrompts = ["\nUSER", "\nSYSTEM", "\nASSISTANT"],
            MaxTokens = 2048
        };

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"Maximum tokens: {inferenceParams.MaxTokens} and the context size is {parameters.ContextSize}.");

        var promptTemplate = """
            Tell me a joke.
            """;

        //var promptTemplate = """
        //    You are a rover on Mars planet with the objective to search for safe passage in your exploration.
        //    The only permited basic actions that a rover can perform are: forward, backward, turn left, turn right and stop.

        //    [Objective]
        //    {{$input}}

        //    [Steps to accomplish the objective]
        //    Proceed following the next steps:
        //    1. Enumerate the permited basic actions.
        //    2. Analyse the image for any potentially dangerous obstacles. Enumerate the obstacles.
        //    3. Assemble an array of actions with reasoning to safely pass detected obstacles, using only the permited basic actions and taking into account the obstacles identified at the previous step.

        //    ASSISTANT:
        //    """;

        /*
              [
                {
                  "action": "<basic action>",
                  "obstacle": "<current obstacle>",
                  "reasoning": "<reasoning of taking the action avoiding the current obstacle>"
                }
              ]         
         */

        ////session.AddSystemMessage("""
        ////    You are a rover on Mars planet and you are searching for safe passage in your exploration.
        ////    The only permited basic actions that a rover can perform are: forward, backward, turn left, turn right and stop.
        ////    """);

        ////session.AddUserMessage(promptTemplate);

        var executionSettings = new OpenAIPromptExecutionSettings
        {
            MaxTokens = 2048,
            Temperature = 0.1f,
        };
        var kernelArguments = new KernelArguments(executionSettings)
        {
            //["input"] = "In front of the rover there is a large hole, in the right side of the rover there is a large boulder. The left side seems to be safe to pass."
            ["input"] = "testers"
        };

        //var promptTemplateFactory = new KernelPromptTemplateFactory();
        //var promptTemplateRenderer = promptTemplateFactory.Create(new PromptTemplateConfig(promptTemplate));
        //var renderedPrompt = await promptTemplateRenderer.RenderAsync(kernel, kernelArguments);
        //Console.WriteLine("-- rendered prompt -------------------------------------------------------");
        //await kernel.PrintRenderedPromptAsync(renderedPrompt, kernelArguments);

        ////var semanticFunction = kernel.CreateFunctionFromPrompt(promptTemplate);



        var handlebarsPlannerOptions = new HandlebarsPlannerOptions { AllowLoops = false };
        var planner = new HandlebarsPlanner(handlebarsPlannerOptions);
        var plan = await planner.CreatePlanAsync(kernel, promptTemplate/*, kernelArguments*/);

        Console.WriteLine("-- plan -------------------------------------------------");
        Console.WriteLine(plan);

        Console.WriteLine("-- response -------------------------------------------------");
        var result = string.Empty;

        var x = await plan.InvokeAsync(kernel/*, kernelArguments*/);
        Console.WriteLine(x);
        Console.WriteLine("-- done -------------------------------------------------");

        return Ok(result);
    }

    [HttpGet("/llamasharp/stepwise")]
    public async Task<IActionResult> GetFunctionCallingStepwisePlannerAsync(string? imagePath, string functionCallingModelAlias)
    {
        var LLM = Models?[functionCallingModelAlias];

        Console.WriteLine("\n===========================================================================");
        Console.WriteLine($"{Path.GetFileNameWithoutExtension(LLM.ModelPath)}");
        Console.WriteLine("---------------------------------------------------------------------------");

        imagePath ??= RoverImagePath;

        NativeLibraryConfig.Instance.WithCuda(true);

        NativeLogConfig.llama_log_set((level, message) =>
        {
            if (message.StartsWith("llama_") ||
                message.StartsWith("llm_") ||
                message.StartsWith("clip_") ||
                message.StartsWith('.'))
                return;
            Console.WriteLine(message.TrimEnd('\n'));
        });

        var parameters = new ModelParams(LLM.ModelPath)
        {
            ContextSize = 4096,
            Seed = 111,
            GpuLayerCount = 8,
        };

        using var model = await LLamaWeights.LoadFromFileAsync(parameters);
        using var context = model.CreateContext(parameters);

        var executor = new InteractiveExecutor(context);

        var builder = Kernel.CreateBuilder();
        ////builder.Services.AddKeyedSingleton<ITextGenerationService>("local-llava", new LLamaSharpTextCompletion(executor));
        builder.Services.AddKeyedSingleton<IChatCompletionService>("local-llama", new LLamaSharpChatCompletion(executor));

        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            //.MinimumLevel.Information()
            //.MinimumLevel.Warning()
            .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
            .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
            .MinimumLevel.Override("System.Net.Http", Serilog.Events.LogEventLevel.Warning)
            .WriteTo.Console()
            .CreateLogger();

        // Add services to the container.
        builder.Services.AddLogging(c => c.AddSerilog(Log.Logger));

        var kernel = builder.Build();

        ////var session = new ChatSession(executor);

        var inferenceParams = new InferenceParams
        {
            Temperature = 0.4f,
            AntiPrompts = ["\nUSER"],
            MaxTokens = 2048
        };

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"Maximum tokens: {inferenceParams.MaxTokens} and the context size is {parameters.ContextSize}.");

        var promptTemplate = """
            Tell me a joke.
            """;

        var executionSettings = new PromptExecutionSettings
        {
            //MaxTokens = 1024,
            //Temperature = 0.1,
            //ResponseFormat = "json_object",
            //StopSequences = ["\nUSER", "\nSYSTEM", "\nASSISTANT"],
        };
        var kernelArguments = new KernelArguments(executionSettings)
        {


            //["input"] = "testers"
        };

        //var promptTemplateFactory = new KernelPromptTemplateFactory();
        //var promptTemplateRenderer = promptTemplateFactory.Create(new PromptTemplateConfig(promptTemplate));
        //var renderedPrompt = await promptTemplateRenderer.RenderAsync(kernel, kernelArguments);
        //Console.WriteLine("-- rendered prompt -------------------------------------------------------");
        //await kernel.PrintRenderedPromptAsync(renderedPrompt, kernelArguments);

        var x = kernel.AutoFunctionInvocationFilters;

        try
        {
            var options = new FunctionCallingStepwisePlannerOptions
            {
                MaxTokens = 1024,
                ExecutionSettings = OpenAIPromptExecutionSettings.FromExecutionSettings(new PromptExecutionSettings { }, 1024)
            };

            var planner = new FunctionCallingStepwisePlanner(options);

            ////var chatHistory = new LLama.Common.ChatHistory();

            var plannerResult = await planner.ExecuteAsync(kernel, promptTemplate);

            Console.WriteLine($"\nFINAL RESULT: {plannerResult.FinalAnswer}");
        }
        catch (Exception ex)
        {
            Log.Error("FAILED with exception: {message}", ex.Message);
        }

        return Ok();
    }
}

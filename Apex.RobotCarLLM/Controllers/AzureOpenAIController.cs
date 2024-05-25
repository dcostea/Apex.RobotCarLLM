using Microsoft.AspNetCore.Mvc;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel;
using Swashbuckle.AspNetCore.Annotations;
using Apex.RobotCarLLM.Helpers;
using System;
using Azure.AI.OpenAI;
using OllamaSharp;
using Serilog;
using System.Runtime.CompilerServices;
using System.Net;

namespace Apex.RobotCarLLM.Controllers;

[ApiController]
[Route("[controller]")]
public class AzureOpenAIController : ControllerBase
{
    private readonly string MotorHelperPlugin = nameof(MotorHelperPlugin);
    private readonly string BreakdownComplexCommands = nameof(BreakdownComplexCommands);
    private readonly string Plugins = nameof(Plugins);

    [HttpGet("/openai/chat/image")]
    [SwaggerOperation(Summary = "Gets the details of the specified model by key.")]
    public async Task<IActionResult> GetChatCompletionWithImageAsync(/*string? imagePath*/)
    {
        var kernel = Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(
                deploymentName: Env.Var("AzureOpenAI:ChatCompletionDeploymentName")!,
                endpoint: Env.Var("AzureOpenAI:Endpoint")!,
                apiKey: Env.Var("AzureOpenAI:ApiKey")!)
            .Build();

        //TODO image content creation from bytes array is buggy
        //imagePath ??= RoverImagePath;
        //var imageBytes = System.IO.File.ReadAllBytes(imagePath);
        //var readOnlyImageBytes = new ReadOnlyMemory<byte>(imageBytes);
        //var imageContent = new ImageContent(readOnlyImageBytes)

        var chatHistory = new ChatHistory("You are an assistant specialised in named entities extraction.");

        chatHistory.AddUserMessage(new ChatMessageContentItemCollection
        {
            new TextContent("""
                Analyze the attached image in details.

                Find a safe path for the rover towards the other side of the crater.
                """),
            new ImageContent(new Uri("https://apexcode.ro/m12.jpeg"))
        });

        var executionSettings = new OpenAIPromptExecutionSettings
        {
            MaxTokens = 4000,
        };

        var kernelArguments = new KernelArguments(executionSettings)
        {
            ["input"] = "Find a safe path for the rover towards the other side of the crater."
        };

        var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
        var content = await chatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings);

        Console.WriteLine($"{content.Role} > {content.Content}");

        return Ok(content.Content);
    }

    [HttpGet("/openai/function_calling/image")]
    public async Task<IActionResult> GetFunctionCallingnWithImageAsync(/*string? imagePath, */bool showChat = false)
    {
        var kernel = Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(
                deploymentName: Env.Var("AzureOpenAI:ChatCompletionDeploymentName")!,
                endpoint: Env.Var("AzureOpenAI:Endpoint")!,
                apiKey: Env.Var("AzureOpenAI:ApiKey")!)
            .Build();

        kernel.ImportPluginFromType<Plugins.MotorCommandsPlugin.MotorCommandsPlugin>();
        kernel.ImportPluginFromPromptDirectory(Path.Combine(Directory.GetCurrentDirectory(), Plugins, MotorHelperPlugin), MotorHelperPlugin);
        kernel.PrintAllPluginsFunctions();

        //TODO image content creation from bytes array is buggy
        //imagePath ??= RoverImagePath;
        //var imageBytes = System.IO.File.ReadAllBytes(imagePath);
        //var readOnlyImageBytes = new ReadOnlyMemory<byte>(imageBytes);
        //var imageContent = new ImageContent(readOnlyImageBytes)

        var chat = kernel.GetRequiredService<IChatCompletionService>();

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("You are a robotic rover on a mission to explore the planet Mars.");
        chatHistory.AddUserMessage(new ChatMessageContentItemCollection
        {
            new ImageContent(new Uri("https://apexcode.ro/m12.jpeg")),
            new TextContent("""
                The attached image is an aerial view of you, the rover in the image.
                Your objective is to analyze the image for obstacles and to identify the path to the distant hills.
                If the objective is too complex for you, break it down into basic motor commands.

                Proceed with the objective!
                """),
        });

        try
        {
            // AutoInvokeKernelFunctions HAS A LIMIT OF MAX 128 CALLS.

            // In order to capture the autoinvoked tools we need to stream the responses

            var executionSettings = new OpenAIPromptExecutionSettings 
            {
                ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions, 
                MaxTokens = 4000 
            };

            var streamingResult = chat.GetStreamingChatMessageContentsAsync(chatHistory, executionSettings, kernel);

            await foreach (var result in streamingResult)
            {
                var openaiMessageContent = result as OpenAIStreamingChatMessageContent;
                var toolCall = openaiMessageContent?.ToolCallUpdate as StreamingFunctionToolCallUpdate;

                if (showChat)
                {
                    if (openaiMessageContent!.Role == AuthorRole.Assistant)
                    {
                        if (toolCall is not null)
                        {
                            Console.Write($"\nTOOL: {toolCall.Name}");
                        }
                        else
                        {
                            Console.WriteLine();
                        }

                        continue;
                    }

                    if (openaiMessageContent?.FinishReason is not null)
                    {
                        Console.WriteLine($"\nFINISH REASON: {openaiMessageContent?.FinishReason}");
                        continue;
                    }
                }

                Console.Write($"{openaiMessageContent?.Content}");
            }
        }
        catch (Exception ex)
        {
            Log.Error("FAILED with exception: {message}", ex.Message);
            return StatusCode((int)HttpStatusCode.ServiceUnavailable);
        }

        //var executionSettings = new OpenAIPromptExecutionSettings
        //{
        //    MaxTokens = 4000,
        //};
        //var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
        //var content = await chatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings);

        return Ok();
    }
}

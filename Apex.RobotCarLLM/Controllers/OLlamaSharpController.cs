using Apex.RobotCarLLM.Models;
using Microsoft.AspNetCore.Mvc;
using OllamaSharp;
using OllamaSharp.Models;
using OllamaSharp.Models.Chat;
using OllamaSharp.Streamer;
using System.Globalization;

namespace Apex.RobotCarLLM.Controllers;

[ApiController]
[Route("[controller]")]
public class OLlamaSharpController : ControllerBase
{
    [HttpGet("/ollamasharp/chat/image")]
    public async Task<IActionResult> GetChatCompletionWithImageAsync(
        ModelSettings modelSettings,
        string prompt,
        string baseModel = LlavaModels.Llava13b,
        bool isStreaming = true,
        string imageName = "Receipt.png", 
        CancellationToken token = default)
    {
        const string CustomModelName = "test_llava";
        const string ModelUri = "http://localhost:11434";

        var modelFileContent = $"""
            FROM {baseModel}
                    
            PARAMETER temperature {modelSettings.Temperature.ToString("F2", CultureInfo.InvariantCulture)}
            PARAMETER top_k {modelSettings.TopK}
            PARAMETER top_p {modelSettings.TopP.ToString("F2", CultureInfo.InvariantCulture)}
            """;

        //var modelFileContent = $"""
        //    FROM {baseModel}

        //    PARAMETER temperature {modelSettings.Temperature.ToString("F2", CultureInfo.InvariantCulture)}
        //    PARAMETER top_k {modelSettings.TopK}
        //    PARAMETER top_p {modelSettings.TopP.ToString("F2", CultureInfo.InvariantCulture)}
        //    PARAMETER num_ctx {modelSettings.NumCtx}
        //    PARAMETER num_gpu {modelSettings.NumGpu}
        //    PARAMETER num_predict {modelSettings.NumPredict}
        //    PARAMETER tfs_z {modelSettings.TfsZ}
        //    PARAMETER seed {modelSettings.Seed}
        //    PARAMETER repeat_last_n {modelSettings.RepeatLastN}
        //    PARAMETER repeat_penalty {modelSettings.RepeatPenalty.ToString("F2", CultureInfo.InvariantCulture)}
        //    """;

        IEnumerable<Message> response = [];
        var ollama = new OllamaApiClient(new Uri(ModelUri));

        // keep reusing the context to keep the chat topic going
        //ConversationContext? context = null;

        try
        {
            Action<CreateStatus> responseHandler = stream => Console.WriteLine($"Stream status: {stream.Status}");

            await ollama.CreateModel(new CreateModelRequest
            {
                ModelFileContent = modelFileContent,
                Name = CustomModelName,
                Stream = isStreaming
            },
            new ActionResponseStreamer<CreateStatus>(responseHandler),
            token);

            ollama.SelectedModel = CustomModelName;

            Console.ForegroundColor = ConsoleColor.DarkGray;
            var modelInfo = await ollama.ShowModelInformation(CustomModelName, token);
            Console.WriteLine($"Model: {modelInfo.Modelfile}");
            Console.ResetColor();

            Action<ChatResponseStream> streamer = stream => Console.Write(stream?.Message?.Content);
            var chat = ollama.Chat(streamer);

            Console.WriteLine($"User > {prompt}");
            Console.WriteLine("Assistant > ");

            /*
             You're an assistant designed to extract entities. Users will attach an image and you'll respond with entities you've extracted from the image as a JSON object. Here's an example of your output format:

            ```json
            {  
               "name": "",
               "company": "",
               "phone_number": ""
            }```


            You are a helpful AI assistant who makes interesting visualizations based on data. 
            You have access to a sandboxed environment for writing and testing code.
            When you are asked to create a visualization you should follow these steps:
            1. Write the code.
            2. Anytime you write new code display a preview of the code to show your work.
            3. Run the code to confirm that it runs.
            4. If the code is successful display the visualization.
            5. If the code is unsuccessful display the error message and try to revise the code and rerun going through the steps from above again.

             
             */

            prompt = """
                You're an assistant designed to extract entities.
                Users will attach an image and you'll respond with entities you've extracted from the image as a JSON object.

                Here's an example of your output format:
                ```json
                {  
                   "name": "",
                   "company": "",
                   "phone_number": ""
                }```
                
                """;

            var imageBytes = System.IO.File.ReadAllBytes(@$"C:\Temp\{imageName}");
            var imagesBase64 = new List<string> { Convert.ToBase64String(imageBytes) };

            if (string.IsNullOrWhiteSpace(modelSettings.SystemPrompt))
            {
                response = await chat.Send(prompt, imagesBase64, token);
            }
            else 
            {
                _ = await chat.SendAs(ChatRole.System, modelSettings.SystemPrompt, token);
                response = await chat.SendAs(ChatRole.User, prompt, token);
            }

            Console.WriteLine();
        }
        finally 
        {
            await ollama.DeleteModel(CustomModelName, token);
        }
        
        return Ok(response);
    }
}

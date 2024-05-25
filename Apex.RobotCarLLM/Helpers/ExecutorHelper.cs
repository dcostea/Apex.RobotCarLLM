using LLama.Common;
using LLama;
using LLama.Native;

namespace Apex.RobotCarLLM.Helpers;

public static class ExecutorHelper
{
    public static InferenceParams? InferenceParams { get; set; }
    public static ModelParams? ModelParams { get; set; }

    public static async Task<string> ExecuteWithSpinnerAsync(this InteractiveExecutor ex, string prompt)
    {
        var result = string.Empty;

        await foreach (var text in ex.InferAsync(prompt, InferenceParams).Spinner())
        {
            Console.Write(text);
            result += text;
        }

        return result;
    }

    public static void InitLlama(string modelPath)
    {
        // setup llama engine
        NativeLibraryConfig.Instance
            .WithCuda(true)
            .WithAvx(NativeLibraryConfig.AvxLevel.Avx512);
        NativeApi.llama_empty_call();
        //LLamaLogCallback l = (level, message) =>
        //{
        //    if (level == LLamaLogLevel.Info && (message.StartsWith("llama_") || message.StartsWith("llm_") || message.StartsWith(".")))
        //        return;
        //    Console.WriteLine(message);
        //};
        //NativeApi.llama_log_set(l);

        ExecutorHelper.InferenceParams = new InferenceParams()
        {
            Temperature = 0.0f,
            //AntiPrompts = new List<string> { "-" },
            MaxTokens = 1024
        };

        ExecutorHelper.ModelParams = new ModelParams(modelPath)
        {
            ContextSize = 512,
            Seed = 1337,
            GpuLayerCount = 100 // default is 20
        };
    }


    public static async IAsyncEnumerable<string> Spinner(this IAsyncEnumerable<string> source)
    {
        var enumerator = source.GetAsyncEnumerator();

        var characters = new[] { '|', '/', '-', '\\' };

        while (true)
        {
            var next = enumerator.MoveNextAsync();

            var (Left, Top) = Console.GetCursorPosition();

            // Keep showing the next spinner character while waiting for "MoveNextAsync" to finish
            var count = 0;
            while (!next.IsCompleted)
            {
                count = (count + 1) % characters.Length;
                Console.SetCursorPosition(Left, Top);
                Console.Write(characters[count]);
                await Task.Delay(100);
            }

            // Clear the spinner character
            Console.SetCursorPosition(Left, Top);
            Console.Write(" ");
            Console.SetCursorPosition(Left, Top);

            if (!next.Result)
                break;
            yield return enumerator.Current;
        }
    }
}


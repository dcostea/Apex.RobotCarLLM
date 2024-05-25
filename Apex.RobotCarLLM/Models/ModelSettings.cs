using System.ComponentModel;

namespace Apex.RobotCarLLM.Models;

public class ModelSettings
{
    /// <summary>
    /// The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
    /// </summary>
    [DefaultValue(0F)]
    public float Temperature { get; init; }

    /// <summary>
    /// Sets the size of the context window used to generate the next token. (Default: 2048)
    /// </summary>
    [DefaultValue(2048)]
    public int NumCtx { get; init; }

    //The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable.
    [DefaultValue(100)]
    public int NumGpu { get; init; }

    /// <summary>
    /// Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
    /// </summary>
    [DefaultValue(-2)]
    public int NumPredict { get; init; }

    /// <summary>
    /// Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
    /// </summary>
    [DefaultValue(1)]
    public int TfsZ { get; init; }

    /// <summary>
    /// Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
    /// </summary>
    [DefaultValue(100)]
    public int TopK { get; init; }

    /// <summary>
    /// Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    /// </summary>
    [DefaultValue(0.4F)]
    public float TopP { get; init; }

    /// <summary>
    /// Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
    /// </summary>
    [DefaultValue(42)]
    public int Seed { get; init; }

    /// <summary>
    /// The SYSTEM instruction specifies the system message to be used in the template, if applicable
    /// </summary>
    public string? SystemPrompt { get; init; }

    /// <summary>
    /// Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
    /// </summary>
    [DefaultValue(64)]
    public int RepeatLastN { get; init; }

    /// <summary>
    /// Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
    /// </summary>
    [DefaultValue(1.1F)]
    public float RepeatPenalty { get; init; }
}

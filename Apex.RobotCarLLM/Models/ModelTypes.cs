namespace Apex.RobotCarLLM.Models;

/// <summary>
/// Enumeration for model types.
/// </summary>
[Flags]
public enum ModelTypes
{
    Llama = 1,
    Llava,
    LlavaLlama = Llava | Llama,
    FunctionCalling,
    LlamaFunctionCalling = Llama | FunctionCalling,
    LlavaLlamaFunctionCalling = LlavaLlama & FunctionCalling
}

using Apex.RobotCarLLM.Controllers;
using Apex.RobotCarLLM.Models;
using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace Apex.RobotCarLLM.Swagger;

public class CustomSwaggerParameterFilter : IOperationFilter
{
    public void Apply(OpenApiOperation operation, OperationFilterContext context)
    {
        if (operation.Parameters == null)
        {
            return;
        }

        var allModels = new LlamaSharpController().Models;

        var modelAlias = operation.Parameters.FirstOrDefault(p => p.Name == "modelAlias");
        if (modelAlias is { Schema.Type: "string" })
        {
            var models = allModels.Select(s => s.Key);
            modelAlias.Description = string.Join(" <br> ", models);
        }

        var videoModelAlias = operation.Parameters.FirstOrDefault(p => p.Name == "videoModelAlias");
        if (videoModelAlias is { Schema.Type: "string" })
        {
            var models = allModels
                .Where(m =>
                    ((m.Value.ModelType & ModelTypes.FunctionCalling) != ModelTypes.FunctionCalling)
                    &&
                    ((m.Value.ModelType & ModelTypes.Llava) == ModelTypes.Llava)
                )
                .Select(s => s.Key);
            videoModelAlias.Description = string.Join(" <br> ", models);
        }

        var functionCallingModelAlias = operation.Parameters.FirstOrDefault(p => p.Name == "functionCallingModelAlias");
        if (functionCallingModelAlias is { Schema.Type: "string" })
        {
            var models = allModels
                .Where(m =>
                    ((m.Value.ModelType & ModelTypes.FunctionCalling) == ModelTypes.FunctionCalling)
                )
                .Select(s => s.Key);
            functionCallingModelAlias.Description = string.Join(" <br> ", models); ;
        }

        var imageFiles = operation.Parameters.FirstOrDefault(p => p.Name == "imagePath");
        if (imageFiles is { Schema.Type: "string" })
        {
            var images = Directory.GetFiles(@"c:\Temp\RoverImages");
            imageFiles.Description = string.Join(" <br> ", images);
        }
    }
}
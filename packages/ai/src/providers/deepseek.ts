import type { Context, Model, SimpleStreamOptions, StreamFunction } from "../types.js";
import type { AssistantMessageEventStream } from "../utils/event-stream.js";
import { type OpenAICompletionsOptions, streamOpenAICompletions } from "./openai-completions.js";

/**
 * DeepSeek provider options.
 * DeepSeek uses the OpenAI-compatible API with additional `extra_body` parameter
 * for features like thinking mode: `extra_body: {thinking: {type: "enabled"}}`.
 */
export type DeepSeekOptions = OpenAICompletionsOptions;

/**
 * Check if thinking mode is enabled for DeepSeek.
 * Thinking mode is enabled if:
 * 1. Model ID contains "reasoner" (e.g., deepseek-reasoner), OR
 * 2. options.extraBody?.thinking?.type === "enabled"
 * 3. options.reasoningEffort is truthy (enables thinking mode with default settings)
 */
function isThinkingModeEnabled(model: Model<"deepseek-completions">, options?: DeepSeekOptions): boolean {
	if (model.id.includes("reasoner")) {
		return true;
	}
	const thinking = options?.extraBody?.thinking as { type?: string } | undefined;
	if (thinking?.type === "enabled") {
		return true;
	}
	// reasoningEffort indicates thinking mode should be enabled
	if (options?.reasoningEffort) {
		return true;
	}
	return false;
}

/**
 * Filter parameters for DeepSeek thinking mode.
 * According to DeepSeek API docs, in thinking mode:
 * - temperature, top_p, presence_penalty, frequency_penalty are ignored
 * - logprobs, top_logprobs are rejected (should cause error)
 * - reasoningEffort should be removed (converted to extraBody.thinking)
 */
function filterThinkingModeParams(options?: DeepSeekOptions): DeepSeekOptions {
	if (!options) {
		return {};
	}

	const filteredOptions = { ...options };

	// Remove parameters that are ignored in thinking mode
	// Note: temperature is already part of StreamOptions, so we delete it
	delete filteredOptions.temperature;

	// Remove reasoningEffort since we'll convert it to extraBody.thinking
	delete filteredOptions.reasoningEffort;

	// Other parameters may be passed through metadata or not at all
	// We'll check for logprobs and top_logprobs in the options object
	const opts = options as any;
	if (opts.logprobs !== undefined || opts.top_logprobs !== undefined) {
		throw new Error("logprobs and top_logprobs are not supported in DeepSeek thinking mode");
	}

	return filteredOptions;
}

/**
 * Stream a completion using DeepSeek's OpenAI-compatible API.
 * Supports thinking mode via `extraBody` option.
 */
export const streamDeepSeek: StreamFunction<"deepseek-completions", DeepSeekOptions> = (
	model: Model<"deepseek-completions">,
	context: Context,
	options?: DeepSeekOptions,
): AssistantMessageEventStream => {
	// Detect thinking mode
	const thinkingMode = isThinkingModeEnabled(model, options);

	// Filter parameters if in thinking mode
	let filteredOptions = options;
	if (thinkingMode) {
		filteredOptions = filterThinkingModeParams(options);

		// Ensure extraBody includes thinking configuration if not already present
		const thinking = filteredOptions.extraBody?.thinking as { type?: string } | undefined;
		if (!thinking?.type) {
			filteredOptions = {
				...filteredOptions,
				extraBody: {
					...filteredOptions.extraBody,
					thinking: { type: "enabled" as const },
				},
			};
		}
	}

	// Convert to openai-completions API for underlying implementation
	const openaiModel = { ...model, api: "openai-completions" as const };
	return streamOpenAICompletions(openaiModel, context, filteredOptions);
};

/**
 * Simplified streaming interface for DeepSeek.
 * Automatically maps `reasoning` option to appropriate thinking configuration.
 */
export const streamSimpleDeepSeek: StreamFunction<"deepseek-completions", SimpleStreamOptions> = (
	model: Model<"deepseek-completions">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	// For simplified interface, we need to map reasoning option to extraBody
	const deepseekOptions: DeepSeekOptions = { ...options };

	if (options?.reasoning) {
		deepseekOptions.extraBody = {
			thinking: { type: "enabled" as const },
		};
	}

	const openaiModel = { ...model, api: "openai-completions" as const };
	return streamOpenAICompletions(openaiModel, context, deepseekOptions);
};

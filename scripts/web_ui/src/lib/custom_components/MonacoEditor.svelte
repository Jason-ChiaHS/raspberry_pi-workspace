<script lang="ts">
	import { onMount } from 'svelte';

	let { code = $bindable('') }: { code?: string } = $props();
	let loading = $state(true);

	// @ts-ignore
	let editor_instance;
	function try_update_code() {
		setTimeout(() => {
			// @ts-ignore
			if (editor_instance && editor_instance.getValue() != code) {
				editor_instance.setValue(code);
			} else {
				try_update_code();
			}
		}, 1000);
	}

	$effect(() => {
		// This is so that code is tracked
		let _ = code;
		// @ts-ignore
		if (editor_instance && editor_instance.getValue() != code) {
			editor_instance.setValue(code);
		} else {
			try_update_code();
		}
	});

	onMount(() => {
		let script_element = document.createElement('script');
		script_element.setAttribute(
			'src',
			'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs/loader.js'
		);
		document.body.appendChild(script_element);

		// Ignore all ts errors since this is vanilla js we are using and injecting
		script_element.onload = () => {
			// @ts-ignore
			require.config({
				paths: {
					vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.52.2/min/vs'
				}
			});
			// @ts-ignore
			require(['vs/editor/editor.main'], () => {
				// @ts-ignore
				editor_instance = monaco.editor.create(document.getElementById('editor'), {
					overviewRulerLanes: 0,
					overviewRulerBorder: false,
					minimap: {
						enabled: false
					},
					lineNumbers: 'off',
					language: 'ini',
					theme: 'vs-dark',
					automaticLayout: true
				});
				loading = false;
				// @ts-ignore
				editor_instance.onDidChangeModelContent(() => {
					// @ts-ignore
					code = editor_instance.getValue();
				});
			});
		};
	});
</script>

<div id="editor" class="h-full w-full"></div>

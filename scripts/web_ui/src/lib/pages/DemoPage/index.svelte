<script lang="ts" module>
	import { type Demo } from '$lib/demos';
	import type { Snippet } from 'svelte';
	export interface DemoPageProps {
		demo: Demo;
		children?: Snippet;
	}
</script>

<script lang="ts">
	import { Loader } from '@lucide/svelte';
	import { onMount } from 'svelte';
	let { demo, children }: DemoPageProps = $props();
	let loading = $state(true);
	let correct_demo = $state(false);
	let running_demo = $state('');
	onMount(() => {
		fetch('/api/running_demo').then(async (res) => {
			running_demo = await res.text();
			if (running_demo === demo.value) {
				correct_demo = true;
			}
			loading = false;
		});
	});
</script>

{#if loading}
	<div class="flex h-full w-full items-center justify-center">
		<Loader class="text-muted-foreground h-[30%] w-[30%] animate-spin" />
	</div>
{:else if correct_demo}
	{@render children?.()}
{:else if running_demo === 'None'}
	<div class="flex h-full flex-col justify-center text-center">
		<p class="text-muted-foreground text-xl">
			No demo is running, click <code
				class="bg-muted relative rounded px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold"
			>
				Start Demo
			</code>
		</p>
	</div>
{:else}
	<div class="flex h-full flex-col justify-center text-center">
		<p class="text-muted-foreground text-xl">This demo is not running.</p>
		<p class="text-muted-foreground text-xl">The current running demo is: {running_demo}</p>
	</div>
{/if}

<script lang="ts">
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from '$lib/components/ui/sonner/index.js';
	import Header from './header.svelte';
	import '../app.css';
	let { children } = $props();
</script>

<ModeWatcher />
<Toaster />
<div class="flex h-dvh flex-col">
	<Header />
	<div class="h-[calc(100dvh-70px)] pt-2 box-border">
		{@render children()}
	</div>
</div>

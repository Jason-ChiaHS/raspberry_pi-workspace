<script lang="ts">
	import CameraConfig from './CameraConfig.svelte';
	import { page } from '$app/state';
	import { demos } from '$lib/demos';
	import StartDemo from './StartDemo.svelte';
	import SelectDemo from './SelectDemo.svelte';
	import DemoConfig from './DemoConfig.svelte';
	import ModeSwitch from '$lib/custom_components/ModeSwitch.svelte';
</script>

<header
	class="flex h-[70px] w-full flex-none flex-wrap items-center gap-3 border-b border-slate-400 px-5 text-center bg-headerBackground text-headerText"
>
	<h1 class="mr-8 text-2xl font-bold">Pentas Vision</h1>
	<div class="flex flex-auto items-center text-center">
		<CameraConfig />
		{#if Object.values(demos).some((demo) => page.url.pathname.includes(demo.path))}
			<DemoConfig />
			<StartDemo />
		{/if}
		<div class="flex flex-auto items-center justify-end gap-3">
			<!-- <ModeSwitch /> -->
			<SelectDemo />
		</div>
	</div>
</header>

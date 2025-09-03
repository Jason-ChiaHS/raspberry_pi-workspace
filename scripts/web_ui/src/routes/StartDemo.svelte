<script lang="ts">
	import LoaderCircle from '@lucide/svelte/icons/loader-circle';
	import { Button } from '$lib/components/ui/button/index';
	import { toast } from 'svelte-sonner';
	import { demos, type Demo } from '$lib/demos';
	import { getPaths } from '$lib/helpers';

	let loading = $state(false);

	let demo = $derived.by(() => {
		return Object.values(demos).find((demo) => getPaths().some((path) => path === demo.value));
	}) as Demo;
</script>

<Button
	variant="link"
	disabled={loading}
	class="text-headerText font-bold"
	onclick={() => {
		loading = true;
		fetch('/api/start_demo', {
			method: 'POST',
			body: JSON.stringify({ value: demo.value }),
			headers: { 'Content-Type': 'application/json' }
		}).then((res) => {
			loading = false;
			if (!res.ok) {
				toast.error('Error starting demo with the given config, please try again');
			} else {
				location.href = demo.path;
			}
		});
	}}
>
	Start Demo
	{#if loading}
		<LoaderCircle class="animate-spin" />
	{:else}
		<LoaderCircle class="hidden animate-spin" />
	{/if}
</Button>

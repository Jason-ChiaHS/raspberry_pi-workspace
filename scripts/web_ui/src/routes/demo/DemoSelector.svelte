<script lang="ts">
	import * as Select from '$lib/components/ui/select/index.js';
	import { Loader } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';
	import { onMount } from 'svelte';
	import MonacoEditor from '$lib/custom_components/MonacoEditor.svelte';
	import Button from '$lib/components/ui/button/button.svelte';

	let value = $state('');
	let demo_config = $state('');

	let loading = $state(true);
	let demos: { name: string; value: string }[] = $state([]);
	onMount(() => {
		loading = true;
		fetch('/api/demos')
			.then(async (res) => {
				demos = await res.json();
				value = demos[0].value;
			})
			.then(get_demo_config)
			.then(() => {
				loading = false;
			});
	});
	const triggerContent = $derived(demos.find((f) => f.value === value)?.name ?? 'Select a Demo');

	async function get_demo_config() {
		loading = true;
		await fetch('/api/demo_config/get', { body: value, method: 'POST' }).then(async (res) => {
			if (!res.ok) {
				toast.error('Error loading demo?');
				return;
			}
			demo_config = await res.text();
		});
	}
</script>

{#if loading}
	<div class="flex h-full w-full items-center justify-center">
		<Loader class="text-muted-foreground h-[30%] w-[30%] animate-spin" />
	</div>
{:else}
	<div class="flex h-full w-full flex-col">
		<p class="mb-1 ml-2 mt-2">Select a Demo</p>

		<div class="mb-2 ml-1 mr-1">
			<Select.Root
				type="single"
				name="favoriteFruit"
				bind:value
				onValueChange={async (_) => {
					get_demo_config().then(() => {
						loading = false;
					});
				}}
			>
				<Select.Trigger>
					{triggerContent}
				</Select.Trigger>
				<Select.Content>
					<Select.Group>
						{#each demos as demo}
							<Select.Item value={demo.value} label={demo.name}>{demo.name}</Select.Item>
						{/each}
					</Select.Group>
				</Select.Content>
			</Select.Root>
		</div>
		<div class="p-1">
			<Button
				onclick={() => {
					fetch('/api/start', {
						method: 'POST',
						body: JSON.stringify({ value: value, config: demo_config }),
						headers: { 'Content-Type': 'application/json' }
					}).then((res) => {
						if (!res.ok) {
							toast.error('Error starting demo with the given config, please try again');
						}
					});
				}}>Start Demo</Button
			>
			<Button
				onclick={() => {
					fetch('/api/start_raspi_cam_srv', { method: 'POST' }).then(() => {
						window.open(`http://${window.location.hostname}:9001`, '_blank');
					});
				}}>Open Cam Srv</Button
			>
			<Button
				onclick={() => {
					fetch('/api/stop', { method: 'POST' });
				}}>Stop Demo</Button
			>
		</div>
		<div class="mb-1 grow">
			<MonacoEditor bind:code={demo_config} />
		</div>
	</div>
{/if}

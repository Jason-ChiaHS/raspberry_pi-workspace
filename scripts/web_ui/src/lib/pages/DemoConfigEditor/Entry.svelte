<script lang="ts" module>
	export type EntrySpec = 'string' | 'number' | 'boolean' | 'list' | { [key: string]: EntrySpec };
	export type Value = { [key: string]: any } | { [key: string]: Value };
	export interface EntryProps {
		value: Value;
		key: string;
		entry_spec: EntrySpec;
	}
</script>

<script lang="ts">
	import Self from './Entry.svelte';
	import { Input } from '$lib/components/ui/input/index';
	import { Label } from '$lib/components/ui/label/index';
	import { Switch } from '$lib/components/ui/switch/index';
	import * as Accordion from '$lib/components/ui/accordion/index';
	import ListEntry from './ListEntry.svelte';

	let { value = $bindable(), key, entry_spec }: EntryProps = $props();
	let actual_spec = typeof entry_spec === 'string';
</script>

{#if actual_spec}
	<div class="my-2 flex items-center gap-1.5">
		<Label class="block">{key}</Label>
		<!-- <p>key: {key}, value: {JSON.stringify(value)}, type: {entry_spec}</p> -->
		{#if entry_spec === 'boolean'}
			<Switch bind:checked={value.value as unknown as boolean} />
		{:else if entry_spec === 'list'}
			<ListEntry bind:value={value.value} />
		{:else if entry_spec === 'string'}
			<Input bind:value={value.value} type="text" />
		{:else}
			<Input bind:value={value.value} type="number" />
		{/if}
	</div>
{:else}
	<Accordion.Root type="single">
		<Accordion.Item>
			<Accordion.Trigger>For {key}</Accordion.Trigger>
			<Accordion.Content>
				{#each Object.entries(entry_spec) as [k, v]}
					<Self entry_spec={v} bind:value={value[k]} key={k} />
				{/each}
			</Accordion.Content>
		</Accordion.Item>
	</Accordion.Root>
{/if}

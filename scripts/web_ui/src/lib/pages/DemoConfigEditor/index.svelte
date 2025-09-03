<script lang="ts" module>
	import { type Demo } from '$lib/demos';

	export type DemoConfigSpecs =
		| { value: string; type: 'string' }
		| { value: number; type: 'number' }
		| { value: boolean; type: 'boolean' }
		| { value: Array<any>; type: 'list' }
		| { [key: string]: DemoConfigSpecs };
	export interface DemoConfigEditorProps {
		demo: Demo;
	}
</script>

<script lang="ts">
	import Entry, { type EntrySpec } from '$lib/pages/DemoConfigEditor/Entry.svelte';
	import { Button } from '$lib/components/ui/button';
	import { Loader } from '@lucide/svelte';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	let { demo }: DemoConfigEditorProps = $props();
	/// Extracts and creates the object needed to create the entry specs
	function to_entry_specs(demo_config_specs: DemoConfigSpecs) {
		// For TS issues
		let entry_specs: { [key: string]: any } = {};
		demo_config_specs = demo_config_specs as { [key: string]: DemoConfigSpecs };
		for (const key in demo_config_specs) {
			if ('type' in demo_config_specs[key]) {
				entry_specs[key] = demo_config_specs[key].type;
			} else {
				entry_specs[key] = to_entry_specs(demo_config_specs[key]);
			}
		}
		return entry_specs;
	}
	let loading = $state(true);
	// We use states for svelte linter to be happy
	let entry_specs_value_default: DemoConfigSpecs = $state({});
	let entry_specs: EntrySpec = $state({});
	let entry_specs_value: DemoConfigSpecs = $state({});
	onMount(() => {
		fetch('/api/demo_config_spec/get', { method: 'POST', body: demo.value }).then(async (res) => {
			entry_specs_value_default = JSON.parse(await res.text());
			entry_specs_value = entry_specs_value_default;
			entry_specs = to_entry_specs(entry_specs_value);
			loading = false;
		});
	});

	function validate_entry_specs_value(in_entry_specs: DemoConfigSpecs): boolean {
		// For TS issues
		let valid = true;
		in_entry_specs = in_entry_specs as { [key: string]: DemoConfigSpecs };
		for (const key in in_entry_specs) {
			if ('type' in in_entry_specs[key]) {
				if (in_entry_specs[key].type === 'list') {
					valid = valid && Array.isArray(in_entry_specs[key].value);
				}
			} else {
				valid = valid && validate_entry_specs_value(in_entry_specs[key]);
			}
		}
		return valid;
	}
	function to_demo_config(in_entry_specs: DemoConfigSpecs): { [key: string]: any } {
		let demo_config: { [key: string]: any } = {};
		in_entry_specs = in_entry_specs as { [key: string]: DemoConfigSpecs };
		for (const key in in_entry_specs) {
			if ('value' in in_entry_specs[key]) {
				demo_config[key] = in_entry_specs[key].value;
			} else {
				demo_config[key] = to_demo_config(in_entry_specs[key]);
			}
		}
		return demo_config;
	}
</script>

{#if loading}
	<div class="flex h-full w-full items-center justify-center">
		<Loader class="text-muted-foreground h-[30%] w-[30%] animate-spin" />
	</div>
{:else}
	<div class="p-4">
		{#each Object.entries(entry_specs as unknown as Object) as [k, v]}
			<Entry
				entry_spec={v}
				bind:value={entry_specs_value[k as keyof DemoConfigSpecs] as any}
				key={k}
			/>
		{/each}

		<Button
			variant="default"
			onclick={() => {
				if (validate_entry_specs_value(entry_specs_value)) {
					let demo_config = to_demo_config(entry_specs_value);
					fetch('/api/demo_config/set', {
						method: 'POST',
						body: JSON.stringify({ config: demo_config, demo: demo.value }),
						headers: { 'Content-Type': 'application/json' }
					}).then((res) => {
						if (!res.ok) {
							toast.error('Error saving config, please try again');
						} else {
							toast.success('Config saved, you can start the demo now');
						}
					});
				} else {
					toast.error('There is an error with your config');
				}
			}}>Save Config</Button
		>
	</div>
{/if}

<script lang="ts" module>
	export interface EntryProps {
		value: any;
	}
</script>

<script lang="ts">
	import { Input } from '$lib/components/ui/input/index';

	let { value = $bindable() }: EntryProps = $props();
	let error = $derived(!Array.isArray(value));
</script>

<div class="flex w-full flex-col">
	<Input
		bind:value={
			() => {
				if (typeof value === 'string') {
					return value;
				} else {
					return JSON.stringify(value);
				}
			},
			(v) => {
				try {
					let vv = JSON.parse(v);
					value = Array.isArray(vv) ? vv : v;
				} catch {
					value = v;
				}
			}
		}
		type="text"
		class={error ? 'border border-red-500' : ''}
	/>
	{#if error}
		<p class="text-sm text-red-600 dark:text-red-500">Should be a list</p>
	{/if}
</div>

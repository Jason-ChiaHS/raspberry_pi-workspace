<script lang="ts">
	import Check from '@lucide/svelte/icons/check';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu/index.js';
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import ChevronUp from '@lucide/svelte/icons/chevron-up';
	import { demos } from '$lib/demos';
	import { getPaths } from '$lib/helpers';

	let link_dropdown_open = $state(false);
	let current_demo = $derived.by(() => {
		return Object.values(demos).find((demo) => getPaths().some((path) => path === demo.value));
	});
</script>

<DropdownMenu.Root bind:open={link_dropdown_open}>
	<DropdownMenu.Trigger>
		<div class="item-center flex text-center font-bold">
			<p onmouseenter={() => (link_dropdown_open = true)}>
				{current_demo ? current_demo.name : 'Select Demo'}
			</p>
			{#if link_dropdown_open}
				<ChevronUp />
			{:else}
				<ChevronDown />
			{/if}
		</div>
	</DropdownMenu.Trigger>
	<DropdownMenu.Content>
		<DropdownMenu.Group>
			<!-- {#each Object.values(demos).filter((demo) => {
				return demo.value === 'gaze-v2';
			}) as demo} -->
			{#each Object.values(demos) as demo}
				<DropdownMenu.Item>
					<span class="flex size-3.5 items-center justify-center font-bold">
						{#if getPaths().some((path) => path === demo.value)}
							<Check class="size-4" />
						{/if}
					</span>
					<a href={demo.path}>{demo.name}</a>
				</DropdownMenu.Item>
			{/each}
		</DropdownMenu.Group>
	</DropdownMenu.Content>
</DropdownMenu.Root>

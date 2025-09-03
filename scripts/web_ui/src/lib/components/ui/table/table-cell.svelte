<script lang="ts">
	import type { HTMLTdAttributes } from "svelte/elements";
	import type { WithElementRef } from "bits-ui";
	import { cn } from "$lib/utils.js";

	interface TableCellProps extends WithElementRef<HTMLTdAttributes>{
		pwidth?: number,
	}

	let {
		ref = $bindable(null),
		class: className,
		pwidth = 4,
		children,
		...restProps
	}: TableCellProps  = $props();
</script>

<td
	bind:this={ref}
	class={cn("align-middle [&:has([role=checkbox])]:pr-0", `p-${pwidth}`, className)}
	{...restProps}
>
	{@render children?.()}
</td>

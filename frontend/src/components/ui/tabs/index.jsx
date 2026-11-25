// src/components/ui/tabs/index.jsx
"use client";

import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";

const Tabs = TabsPrimitive.Root;
const TabsList = TabsPrimitive.List;
const TabsTrigger = TabsPrimitive.Trigger;
const TabsContent = TabsPrimitive.Content;

export { Tabs, TabsList, TabsTrigger, TabsContent };

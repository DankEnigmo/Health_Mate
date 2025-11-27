"use client";

import React from "react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  CalendarDays,
  BellRing,
  Settings,
  Camera,
} from "lucide-react"; // Removed Pill, PlusCircle icons

const navItems = [
  {
    title: "Dashboard",
    href: "/caregiver/dashboard",
    icon: LayoutDashboard,
  },
  // Removed Medication and Add Medication tabs
  {
    title: "Calendar",
    href: "/caregiver/calendar",
    icon: CalendarDays,
  },
  {
    title: "Alerts",
    href: "/caregiver/alerts",
    icon: BellRing,
  },
  {
    title: "Fall Detection",
    href: "/caregiver/fall-detection",
    icon: Camera,
  },
  {
    title: "Settings",
    href: "/caregiver/settings",
    icon: Settings,
  },
];

const SidebarNav: React.FC = () => {
  return (
    <nav className="flex flex-col gap-2 p-4">
      <h2 className="text-2xl font-bold mb-4 text-sidebar-primary">Health Mate</h2>
      {navItems.map((item) => (
        <NavLink
          key={item.href}
          to={item.href}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sidebar-foreground transition-all hover:text-sidebar-primary",
              isActive && "bg-sidebar-accent text-sidebar-accent-foreground hover:text-sidebar-accent-foreground",
            )
          }
        >
          <item.icon className="h-5 w-5" />
          {item.title}
        </NavLink>
      ))}
    </nav>
  );
};

export default SidebarNav;
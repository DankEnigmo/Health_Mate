"use client";

import React from "react";
import { Link, useLocation } from "react-router-dom";
import { Home, CalendarDays, MessageSquareText, LogOut, Pill } from "lucide-react";
import { cn } from "@/lib/utils";
import EmergencyButton from "./EmergencyButton";
import { useSupabase } from "@/components/SessionContextProvider";
import { Button } from "@/components/ui/button";

interface PatientLayoutProps {
  children: React.ReactNode;
}

const navItems = [
  {
    title: "Message Pad",
    href: "/patient/message-input",
    icon: MessageSquareText,
  },
  {
    title: "Health Overview",
    href: "/patient/dashboard",
    icon: Home,
  },
  {
    title: "Medications",
    href: "/patient/medications",
    icon: Pill,
  },
  {
    title: "Calendar",
    href: "/patient/calendar",
    icon: CalendarDays,
  },
];

const PatientLayout: React.FC<PatientLayoutProps> = ({ children }) => {
  const location = useLocation();
  const { supabase } = useSupabase();

  const handleLogout = async () => {
    await supabase.auth.signOut();
    // Redirection will be handled by ProtectedRoute or RoleSelectionPage
  };

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      {/* Patient-friendly Header/Navigation */}
      <header className="w-full bg-primary text-primary-foreground p-4 shadow-md flex justify-between items-center">
        <h1 className="text-2xl font-bold rubik-doodle-shadow-regular">Health Mate Patient</h1>
        <nav className="flex items-center gap-4">
          {navItems.map((item) => (
            <Link
              key={item.href}
              to={item.href}
              className={cn(
                "flex items-center gap-2 text-primary-foreground hover:text-white transition-colors",
                location.pathname === item.href && "font-semibold underline"
              )}
            >
              <item.icon className="h-5 w-5" />
              <span className="hidden sm:inline">{item.title}</span>
            </Link>
          ))}
          <Button variant="ghost" size="icon" onClick={handleLogout} className="text-primary-foreground hover:text-white">
            <LogOut className="h-5 w-5" />
            <span className="sr-only">Logout</span>
          </Button>
        </nav>
      </header>

      {/* Main content area */}
      <main className="flex-1 p-4 md:p-6 lg:p-8 overflow-auto">
        {children}
      </main>

      {/* Emergency SOS Button */}
      <EmergencyButton />
    </div>
  );
};

export default PatientLayout;
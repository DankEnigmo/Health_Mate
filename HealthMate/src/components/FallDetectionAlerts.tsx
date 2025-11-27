"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BellRing, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";

// Mock data for demonstration purposes, now for a single patient
const mockAlerts = [
  {
    id: "1",
    patientName: "Walter White", // Consistent patient name
    timestamp: "2025-10-27T10:30:00Z",
    status: "New",
    location: "Living Room",
  },
  {
    id: "2",
    patientName: "Walter White", // Consistent patient name
    timestamp: "2025-10-27T09:15:00Z",
    status: "Reviewed",
    location: "Bedroom",
  },
  {
    id: "3",
    patientName: "Walter White", // Consistent patient name
    timestamp: "2025-10-26T18:00:00Z",
    status: "Resolved",
    location: "Kitchen",
  },
];

interface FallDetectionAlertsProps {
  limit?: number; // Optional prop to limit the number of alerts displayed
  showTitle?: boolean; // Optional prop to show a title for the section
}

const FallDetectionAlerts: React.FC<FallDetectionAlertsProps> = ({ limit, showTitle = true }) => {
  const alertsToDisplay = limit ? mockAlerts.slice(0, limit) : mockAlerts;

  return (
    <div className="space-y-4">
      {showTitle && <h2 className="text-2xl font-bold">Fall Detection Alerts</h2>}
      {alertsToDisplay.length === 0 ? (
        <p className="text-muted-foreground">No fall detection alerts to display.</p>
      ) : (
        <div className="grid gap-4">
          {alertsToDisplay.map((alert) => (
            <Card key={alert.id} className={cn(
              alert.status === "New" && "border-destructive ring-destructive ring-1",
              "hover:shadow-md transition-shadow"
            )}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-lg font-semibold flex items-center gap-2">
                  <BellRing className="h-5 w-5 text-destructive" />
                  Fall Detected!
                </CardTitle>
                <span className={cn(
                  "px-2 py-1 text-xs font-medium rounded-full",
                  alert.status === "New" && "bg-destructive text-destructive-foreground",
                  alert.status === "Reviewed" && "bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100",
                  alert.status === "Resolved" && "bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100"
                )}>
                  {alert.status}
                </span>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  <User className="h-4 w-4" /> Patient: <span className="font-medium text-foreground">{alert.patientName}</span>
                </p>
                <p className="text-sm text-muted-foreground">
                  Location: <span className="font-medium text-foreground">{alert.location}</span>
                </p>
                <p className="text-sm text-muted-foreground">
                  Time: <span className="font-medium text-foreground">{new Date(alert.timestamp).toLocaleString()}</span>
                </p>
                <Button variant="outline" size="sm" className="mt-2">
                  View Details
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
      {limit && mockAlerts.length > limit && (
        <div className="text-center mt-4">
          <Button asChild variant="link">
            <Link to="/alerts">View All Alerts</Link>
          </Button>
        </div>
      )}
    </div>
  );
};

export default FallDetectionAlerts;
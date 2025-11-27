"use client";

import React from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const FrontPage: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-primary to-primary-foreground p-4">
      <Card className="w-full max-w-md text-center shadow-lg">
        <CardHeader>
          <CardTitle className="text-4xl font-bold text-primary rubik-doodle-shadow-regular">Health Mate</CardTitle>
          <p className="text-lg text-muted-foreground mt-2">
            Your trusted companion for health and care.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Button asChild className="w-full h-12 text-xl">
            <Link to="/auth">Login</Link>
          </Button>
          <Button asChild variant="outline" className="w-full h-12 text-xl">
            <Link to="/auth?view=signup">Sign Up</Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default FrontPage;
import { Component, ElementRef, ViewChild, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatService } from './chat.service';

interface Message {
  from: 'user' | 'bot';
  text: string;
  agents?: string[];
  error?: boolean;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  private chat = inject(ChatService);
  @ViewChild('scrollArea') scrollArea?: ElementRef<HTMLDivElement>;

  // Roles the backend understands (RAG and Driver agents filter by role).
  roles = [
    { value: 'dispatcher', label: 'Dispatcher' },
    { value: 'finance', label: 'Finance' },
    { value: 'compliance', label: 'Compliance' },
    { value: 'ops_manager', label: 'Operations Manager' },
    { value: 'driver_manager', label: 'Driver Manager' },
  ];
  role = 'dispatcher';

  // One thread ID per conversation: the backend's memory is keyed on it.
  threadId = this.newThreadId();
  messages: Message[] = [];
  question = '';
  loading = false;

  // Quick actions: the demo questions, one click each (Design Doc Section 10).
  quickActions = [
    { label: 'Why did D102 settlement drop?', question: 'Why did D102 settlement drop?' },
    { label: 'Why was the fuel cost so high?', question: 'Why was the fuel cost so high?' },
    { label: 'Why is shipment S10023 delayed?', question: 'Why is shipment S10023 delayed?' },
    { label: 'Why is D104 safety score dropping?', question: 'Why is D104 safety score dropping?' },
    { label: 'Why did D105 escrow balance drop?', question: 'Why did D105 escrow balance drop?' },
    { label: 'Tell me about D103', question: 'Tell me about D103' },
    { label: 'What is the escrow withdrawal policy?', question: 'What is the escrow withdrawal policy?' },
    { label: 'Terminate D102', question: 'Terminate D102' },
  ];

  // Friendly names for the agent-trace chips.
  agentLabels: Record<string, string> = {
    settlement: 'Settlement',
    performance: 'Performance',
    escrow: 'Escrow',
    shipment: 'Shipment / Risk',
    fuel: 'Fuel',
    driver: 'Driver',
    rag: 'Policy (RAG)',
  };

  send(text: string = this.question): void {
    const q = text.trim();
    if (!q || this.loading) return;

    this.messages.push({ from: 'user', text: q });
    this.question = '';
    this.loading = true;
    this.scrollToBottom();

    this.chat.ask(q, this.role, this.threadId).subscribe({
      next: (res) => {
        this.messages.push({ from: 'bot', text: this.clean(res.answer), agents: res.agents_used });
        this.loading = false;
        this.scrollToBottom();
      },
      error: () => {
        this.messages.push({
          from: 'bot',
          text: "Sorry, I couldn't reach the LogiMind server. Please check that the backend is running.",
          error: true,
        });
        this.loading = false;
        this.scrollToBottom();
      },
    });
  }

  newConversation(): void {
    this.messages = [];
    this.threadId = this.newThreadId();
  }

  agentLabel(name: string): string {
    return this.agentLabels[name] ?? name;
  }

  // The LLM sometimes uses **bold** markers; show plain text instead.
  private clean(text: string): string {
    return text.replace(/\*\*/g, '');
  }

  private newThreadId(): string {
    return 'ui-' + Date.now().toString(36);
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const el = this.scrollArea?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }
}
